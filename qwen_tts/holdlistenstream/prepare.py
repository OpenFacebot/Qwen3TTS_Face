"""Rebuild Qwen streams from the preserved Seamless ZIPs and frozen codec weights."""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import soundfile as sf

from .files import atomic_write, digest, file_hash, model_identity, read_contract, read_json, write_json
from .motion import PREPROCESS, prepare_motion
from .protocol import VARIANTS, annotation_slots, validate_slots


class Archives:
    def __init__(self, capacity=8):
        self.open = OrderedDict()
        self.capacity = capacity

    def get(self, path):
        if path not in self.open:
            self.open[path] = zipfile.ZipFile(path)
        self.open.move_to_end(path)
        while len(self.open) > self.capacity:
            self.open.popitem(last=False)[1].close()
        return self.open[path]

    def close(self):
        for z in self.open.values():
            z.close()
        self.open.clear()


def wav_from_bytes(data):
    from scipy.signal import resample_poly
    from math import gcd
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x.mean(1)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite waveform")
    if sr != 24000:
        g = gcd(sr, 24000)
        x = resample_poly(x, 24000 // g, sr // g).astype(np.float32)
    return x


def index_sources(old_cache, raw_root):
    """Copy only identities/spans from old caches; no old token data is reused."""
    import pyarrow.parquet as pq
    records, seen, dialogues = [], set(), {}
    archives = Archives()
    try:
        for split in ("train", "dev"):
            lists = sorted((Path(old_cache) / split / "parquet/holdlistenstream").glob("*/data.list"))
            if not lists:
                raise ValueError(f"Missing audited {split} source list")
            for line in lists[0].read_text().splitlines():
                if not line.strip():
                    continue
                shard = Path(line.strip())
                if not shard.is_absolute():
                    shard = lists[0].parent / shard
                columns = ["utt", "dialogue_id", "source_wav", "start_frame", "motion_num_frames"]
                for item in pq.read_table(shard, columns=columns).to_pylist():
                    utt = item["utt"]
                    if utt in seen or Path(utt).name != utt:
                        raise ValueError(f"Duplicate or invalid identity: {utt}")
                    seen.add(utt)
                    dialogue = item["dialogue_id"]
                    if dialogues.setdefault(dialogue, split) != split:
                        raise ValueError("Dialogue leakage between splits")
                    source = Path(item["source_wav"].split("/data_arkit/", 1)[1])
                    archive = (Path(raw_root) / source.parent).with_suffix(".zip").resolve()
                    person = source.stem
                    z = archives.get(str(archive))
                    names = [person + suffix for suffix in (".wav", ".json", "_arkit61.npy")]
                    members = {name: {"bytes": z.getinfo(name).file_size, "crc32": z.getinfo(name).CRC} for name in names}
                    start, frames = item["start_frame"], item["motion_num_frames"]
                    if start % 10 or frames % 10 or frames < 50:
                        raise ValueError(f"Invalid audited span: {utt}")
                    stat = archive.stat()
                    records.append(dict(utt=utt, split=split, dialogue_id=dialogue, person=person,
                                        speaker=person.rsplit("_", 1)[-1], archive=str(archive), members=members,
                                        archive_size=stat.st_size, archive_mtime_ns=stat.st_mtime_ns,
                                        start_frame=start, frames=frames, blocks=frames // 10))
    finally:
        archives.close()
    return sorted(records, key=lambda x: (x["split"], x["archive"], x["person"], x["start_frame"]))


def check_source(record, archives):
    path = Path(record["archive"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (record["archive_size"], record["archive_mtime_ns"]):
        raise ValueError(f"Source ZIP changed: {path}")
    z = archives.get(str(path))
    for name, expected in record["members"].items():
        info = z.getinfo(name)
        if (info.file_size, info.CRC) != (expected["bytes"], expected["crc32"]):
            raise ValueError(f"Source member changed: {name}")
    return z


def initialize(args):
    from .codecs import load_face
    output = Path(args.output)
    contract_path = output / "contract.json"
    if contract_path.exists():
        if args.rebuild == "all":
            raise ValueError("--rebuild all requires a new output directory; existing builds are never overwritten")
        contract = read_contract(contract_path)
        if set(args.codecs) != set(contract["codecs"]):
            raise ValueError("Requested codecs differ from the existing build")
        if model_identity(args.model) != contract["model_files"]:
            raise ValueError("Qwen model/tokenizer snapshot changed")
        for variant, asset in contract["codecs"].items():
            if file_hash(asset["path"]) != asset["sha256"]:
                raise ValueError(f"Codec weights changed: {variant}")
        return contract
    records = read_json(args.source_manifest) if args.source_manifest else index_sources(args.old_cache, args.raw_root)
    seen, dialogue_splits = set(), {}
    for record in records:
        if record["split"] not in ("train", "dev"):
            raise ValueError("Source manifest requires train/dev splits")
        for key in ("utt", "person", "speaker"):
            value = record[key]
            if not isinstance(value, str) or not value or Path(value).name != value or value in (".", "..") or "\\" in value:
                raise ValueError(f"Unsafe source manifest identity: {key}")
        if record["utt"] in seen or dialogue_splits.setdefault(record["dialogue_id"], record["split"]) != record["split"]:
            raise ValueError("Duplicate stream or dialogue leakage in source manifest")
        seen.add(record["utt"])
    codecs = {}
    for variant in args.codecs:
        path = Path(args.codec_root) / variant / "checkpoints/epoch_005.pt"
        model, _, asset = load_face(path, variant)
        codecs[variant] = asset
        del model
    model_files = model_identity(args.model)
    contract = dict(version=1, algorithm="qwen-holdlistenstream-v1", model=str(Path(args.model).resolve()),
                    model_files=model_files, codecs=codecs, preprocessing=PREPROCESS,
                    clock=dict(seconds=.4, audio_hz=12.5, motion_hz=25, text_slots=4, release_delay=.8),
                    source_manifest_sha256=digest(records), counts={s: sum(r["split"] == s for r in records) for s in ("train", "dev")})
    contract["fingerprint"] = digest(contract)
    write_json(output / "sources.json", records)
    write_json(contract_path, contract)
    print(json.dumps({"stage": "index", "counts": contract["counts"], "fingerprint": contract["fingerprint"]}), flush=True)
    return contract


def verified(path, fingerprint):
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not receipt.exists():
        return False
    try:
        item = read_json(receipt)
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
        return False
    if not isinstance(item, dict):
        return False
    return item.get("fingerprint") == fingerprint and item.get("sha256") == file_hash(path)


def save_npz(path, fingerprint, arrays, details):
    atomic_write(path, lambda f: np.savez_compressed(f, **arrays))
    write_json(str(path) + ".json", dict(fingerprint=fingerprint, sha256=file_hash(path), **details))


def reuse_common(args, contract, records):
    """Copy validated codec-independent payloads into a separate dataset build."""
    source, output = Path(args.reuse_common_from), Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("Reuse requires a separate output dataset")
    parent = read_contract(source / "contract.json")
    report = read_json(source / "validation.json")
    if report["status"] != "complete" or report["fingerprint"] != parent["fingerprint"]:
        raise ValueError("Source dataset has not passed complete validation")
    for key in ("version", "algorithm", "model_files", "preprocessing", "clock", "source_manifest_sha256", "counts"):
        if parent[key] != contract[key]:
            raise ValueError(f"Cannot reuse common data: incompatible {key}")
    if digest(read_json(source / "sources.json")) != digest(records) or digest(records) != parent["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    refs = read_json(source / "references.json")
    expected_refs = {r["split"] + "-" + r["speaker"] for r in records}
    if set(refs) != expected_refs:
        raise ValueError("Reference participants differ from the source manifest")
    variant = next(iter(parent["codecs"]))
    expected, index_hashes = {}, {}
    for split in ("train", "dev"):
        index = source / variant / (split + ".jsonl")
        index_hashes[split] = file_hash(index)
        if index_hashes[split] != report["variants"][variant][split]["index_sha256"]:
            raise ValueError("Source dataset index checksum mismatch")
        rows = [json.loads(line) for line in index.read_text().splitlines() if line]
        by_id = {r["utt"]: r for r in rows}
        selected = [r for r in records if r["split"] == split]
        if len(by_id) != len(rows) or set(by_id) != {r["utt"] for r in selected}:
            raise ValueError("Source dataset index identities differ")
        for record in selected:
            row = by_id[record["utt"]]
            key = split + "-" + record["speaker"]
            common, reference = f"common/{split}/{record['utt']}.npz", f"references/{key}.npz"
            if row["blocks"] != record["blocks"] or row["common"] != common or row["reference"] != reference or refs[key] != reference:
                raise ValueError("Source cache paths or temporal spans differ")
            for name, relative in (("common", common), ("reference", reference)):
                checksum = row["checksums"][name]
                if expected.setdefault(relative, checksum) != checksum:
                    raise ValueError("Conflicting source cache checksums")
    for number, (relative, checksum) in enumerate(sorted(expected.items()), 1):
        src, target = source / relative, output / relative
        receipt = read_json(str(src) + ".json")
        if receipt.get("fingerprint") != parent["fingerprint"] or receipt.get("sha256") != checksum or file_hash(src) != checksum:
            raise ValueError(f"Source cache checksum mismatch: {src}")
        if not verified(target, contract["fingerprint"]) or file_hash(target) != checksum:
            def copy_payload(stream):
                with src.open("rb") as incoming:
                    shutil.copyfileobj(incoming, stream, length=8 << 20)
            atomic_write(target, copy_payload)
            if file_hash(target) != checksum:
                raise ValueError(f"Copied cache checksum mismatch: {target}")
            receipt.update(fingerprint=contract["fingerprint"],
                           reused_from=dict(path=str(source.resolve()), fingerprint=parent["fingerprint"]))
            write_json(str(target) + ".json", receipt)
        if number % 1000 == 0:
            print(json.dumps(dict(stage="reuse_common", copied_or_verified=number, total=len(expected))), flush=True)
    write_json(output / "references.json", refs)
    write_json(output / "common_reuse.json", dict(source=str(source.resolve()), source_fingerprint=parent["fingerprint"],
               index_sha256=index_hashes, common_streams=len(records), references=len(refs), artifacts=len(expected)))
    print(json.dumps(dict(stage="reuse_common", status="complete", common_streams=len(records), references=len(refs))), flush=True)


def prepare_references(args, contract, records):
    from .codecs import load_speaker, speaker_vector
    output, fingerprint = Path(args.output), contract["fingerprint"]
    candidates, archives = {}, Archives()
    for record in records:
        z = check_source(record, archives)
        annotation = json.loads(z.read(record["person"] + ".json"))
        key = record["split"] + "-" + record["speaker"]
        for interval in annotation["metadata:vad"]:
            start, end = interval["start"], interval["end"]
            if not np.isfinite([start, end]).all() or start < 0 or end <= start:
                raise ValueError("Invalid reference speech interval")
            candidate = (end - start, record["person"], start, end, record)
            if key not in candidates or candidate[:3] > candidates[key][:3]:
                candidates[key] = candidate
    expected = {r["split"] + "-" + r["speaker"] for r in records}
    if expected - candidates.keys():
        raise ValueError(f"No voiced reference for {sorted(expected - candidates.keys())}")
    model = load_speaker(contract["model"], args.device)
    references = {}
    for key, (duration, _, start, end, record) in sorted(candidates.items()):
        path = output / "references" / (key + ".npz")
        if not verified(path, fingerprint):
            raw = check_source(record, archives).read(record["person"] + ".wav")
            waveform = wav_from_bytes(raw)
            center = (start + end) / 2
            left, right = max(start, center - 3), min(end, center + 3)
            ref = waveform[round(left * 24000):round(right * 24000)]
            if len(ref) < 1024:
                raise ValueError(f"Reference too short for speaker encoder: {key}")
            vector = speaker_vector(model, ref)
            save_npz(path, fingerprint, dict(waveform=ref, speaker=vector),
                     dict(archive=record["archive"], member=record["person"] + ".wav", start=left,
                          seconds=len(ref) / 24000, source_sha256=hashlib.sha256(raw).hexdigest(), short=len(ref) < 72000))
        references[key] = str(path.relative_to(output))
    archives.close()
    write_json(output / "references.json", references)
    print(json.dumps({"stage": "references", "count": len(references)}), flush=True)


def encode_records(args, contract, records):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    from qwen_tts import Qwen3TTSTokenizer
    from .codecs import encode_audio_batch, encode_face, load_face
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output, fingerprint = Path(args.output), contract["fingerprint"]
    text = AutoTokenizer.from_pretrained(contract["model"], fix_mistral_regex=True)
    audio = Qwen3TTSTokenizer.from_pretrained(str(Path(contract["model"]) / "speech_tokenizer"),
                                            device_map=args.device, torch_dtype=torch.float32, attn_implementation="sdpa")
    audio.model.eval().requires_grad_(False)
    faces = {}
    for name, asset in contract["codecs"].items():
        model, _, actual = load_face(asset["path"], name, args.device)
        if actual["sha256"] != asset["sha256"]:
            raise ValueError(f"Codec weights changed during loading: {name}")
        faces[name] = model
    # Duration buckets bound padding waste; complete batches have stable worker
    # assignment. CPU ZIP reads/filtering overlap the preceding GPU batch.
    ordered = sorted(records, key=lambda r: (r["start_frame"] + r["frames"], r["archive"], r["person"]))
    batches = [ordered[i:i + args.audio_batch_size] for i in range(0, len(ordered), args.audio_batch_size)][args.worker::args.workers]
    if args.max_recordings:
        selected = [r for batch in batches for r in batch][:args.max_recordings]
        batches = [selected[i:i + args.audio_batch_size] for i in range(0, len(selected), args.audio_batch_size)]
    total = sum(map(len, batches))
    start_time, number, media_seconds = time.monotonic(), 0, 0

    def prepare_cpu(record):
        utt, person = record["utt"], record["person"]
        common = output / "common" / record["split"] / (utt + ".npz")
        face_paths = {name: output / name / record["split"] / (utt + ".npz") for name in faces}
        ready = verified(common, fingerprint)
        face_ready = {name: verified(path, fingerprint) for name, path in face_paths.items()}
        job = dict(record=record, common=common, face_paths=face_paths, ready=ready, face_ready=face_ready)
        archives = Archives(1)
        try:
            z = check_source(record, archives)
            if not ready or not all(face_ready.values()):
                raw_motion = z.read(person + "_arkit61.npy")
                motion, diagnostics = prepare_motion(np.load(io.BytesIO(raw_motion), allow_pickle=False))
                left, end = record["start_frame"], record["start_frame"] + record["frames"]
                if end > len(motion):
                    raise ValueError(f"Audited motion span exceeds raw duration: {utt}")
                source_hashes = {"motion": hashlib.sha256(raw_motion).hexdigest()}
                job.update(motion=motion, diagnostics=diagnostics, source_hashes=source_hashes, left=left, end=end)
                if not ready:
                    raw_audio, raw_annotation = z.read(person + ".wav"), z.read(person + ".json")
                    waveform = wav_from_bytes(raw_audio)
                    if abs(len(waveform) / 960 - len(motion)) > 3:
                        raise ValueError(f"Audio/motion drift exceeds 3 motion frames: {utt}")
                    if end * 960 > len(waveform):
                        raise ValueError(f"Audited audio span exceeds raw duration: {utt}")
                    source_hashes.update(audio=hashlib.sha256(raw_audio).hexdigest(), annotation=hashlib.sha256(raw_annotation).hexdigest())
                    job.update(waveform=waveform[:end * 960], annotation=json.loads(raw_annotation))
            return job
        finally:
            archives.close()

    with ThreadPoolExecutor(max_workers=args.cpu_workers) as pool, ThreadPoolExecutor(max_workers=args.cpu_workers) as writers:
        pending = [pool.submit(prepare_cpu, r) for r in batches[0]] if batches else []
        for batch_index in range(len(batches)):
            jobs = [f.result() for f in pending]
            writes = []
            pending = [pool.submit(prepare_cpu, r) for r in batches[batch_index + 1]] if batch_index + 1 < len(batches) else []
            fresh = [job for job in jobs if not job["ready"]]
            codes = encode_audio_batch(audio, [job["waveform"] for job in fresh]) if fresh else []
            for job, audio_codes in zip(fresh, codes):
                record, left, end = job["record"], job["left"], job["end"]
                audio_codes = audio_codes[left // 2:end // 2]
                slots = np.asarray(annotation_slots(job["annotation"], len(job["motion"]),
                                   lambda s: text.encode(s, add_special_tokens=False)), dtype=np.int32)[left // 10:end // 10]
                validate_slots(slots, len(text))
                if audio_codes.shape != (record["blocks"] * 5, 16) or audio_codes.max() >= 2048:
                    raise ValueError(f"Invalid Qwen audio codes: {record['utt']}, {audio_codes.shape}")
                writes.append(writers.submit(save_npz, job["common"], fingerprint, dict(audio=audio_codes, text=slots),
                         dict(utt=record["utt"], blocks=record["blocks"], source_sha256=job["source_hashes"],
                              tail_motion_frames=len(job["motion"]) - end, diagnostics=job["diagnostics"],
                              processing=dict(audio_batch_size=len(fresh), chunk_seconds=30, dtype="float32", tf32=False,
                                              torch=str(torch.__version__), numpy=np.__version__))))
            for job in jobs:
                record = job["record"]
                for name, model in faces.items():
                    if job["face_ready"][name]:
                        continue
                    codes = encode_face(model, job["motion"][:job["end"]])[job["left"] // 5:job["end"] // 5]
                    if codes.shape != (record["blocks"] * 2, len(model.sizes)):
                        raise ValueError(f"Invalid face code shape: {record['utt']}")
                    for book, size in enumerate(model.sizes):
                        if codes[:, book].max() >= size:
                            raise ValueError("Face code outside vocabulary")
                    writes.append(writers.submit(save_npz, job["face_paths"][name], fingerprint, dict(face=codes),
                             dict(utt=record["utt"], blocks=record["blocks"], source_sha256=job["source_hashes"]["motion"],
                                  codec_sha256=contract["codecs"][name]["sha256"], diagnostics=job["diagnostics"])))
                number += 1
                media_seconds += record["frames"] / 25
            for future in writes:
                future.result()
            if batch_index % 5 == 0 or number == total:
                status = dict(stage="encode", worker=args.worker, completed=number, assigned=total,
                              media_seconds=media_seconds, seconds=round(time.monotonic() - start_time, 2),
                              last=jobs[-1]["record"]["utt"])
                write_json(output / f"worker-{args.worker}.json", status)
                print(json.dumps(status), flush=True)


def validate_build(args, contract, records):
    output, fingerprint = Path(args.output), contract["fingerprint"]
    refs = read_json(output / "references.json")
    from transformers import AutoTokenizer
    vocab_size = len(AutoTokenizer.from_pretrained(contract["model"], fix_mistral_regex=True))
    reports, diagnostics = {}, dict(streams_with_nonfinite_motion=0, raw_nonfinite_values=0, max_tail_motion_frames=0)
    short_references = {}
    for key, relative in refs.items():
        path = output / relative
        if not verified(path, fingerprint):
            raise ValueError(f"Invalid reference artifact: {path}")
        receipt = read_json(str(path) + ".json")
        if receipt["short"]:
            short_references[key] = receipt["seconds"]
    for variant, asset in contract["codecs"].items():
        reports[variant] = {}
        for split in ("train", "dev"):
            rows, blocks = [], 0
            for record in records:
                if record["split"] != split:
                    continue
                common = output / "common" / split / (record["utt"] + ".npz")
                face = output / variant / split / (record["utt"] + ".npz")
                ref = output / refs[split + "-" + record["speaker"]]
                for path in (common, face, ref):
                    if not verified(path, fingerprint):
                        raise ValueError(f"Missing, stale or corrupt artifact: {path}")
                if variant == next(iter(contract["codecs"])):
                    receipt = read_json(str(common) + ".json")
                    bad = receipt["diagnostics"]["raw_nonfinite"]
                    diagnostics["streams_with_nonfinite_motion"] += int(bad > 0)
                    diagnostics["raw_nonfinite_values"] += bad
                    diagnostics["max_tail_motion_frames"] = max(diagnostics["max_tail_motion_frames"], receipt["tail_motion_frames"])
                with np.load(common, allow_pickle=False) as a, np.load(face, allow_pickle=False) as f, np.load(ref, allow_pickle=False) as r:
                    n, sizes = record["blocks"], asset["geometry"]["sizes"]
                    validate_slots(a["text"], vocab_size)
                    if a["text"].shape != (n, 4) or a["audio"].shape != (n * 5, 16) or f["face"].shape != (n * 2, len(sizes)):
                        raise ValueError("Cache shape mismatch")
                    if a["audio"].max() >= 2048 or any(f["face"][:, i].max() >= v for i, v in enumerate(sizes)):
                        raise ValueError("Cache code range mismatch")
                    if r["speaker"].shape != (2048,) or not np.isfinite(r["speaker"]).all():
                        raise ValueError("Invalid speaker cache")
                rows.append(dict(utt=record["utt"], blocks=n, common=str(common.relative_to(output)),
                                 face=str(face.relative_to(output)), reference=str(ref.relative_to(output)),
                                 checksums={key: read_json(str(path) + ".json")["sha256"]
                                            for key, path in (("common", common), ("face", face), ("reference", ref))}))
                blocks += n
                if len(rows) % 1000 == 0:
                    print(json.dumps(dict(stage="validate", codec=variant, split=split,
                                          validated=len(rows), expected=contract["counts"][split])), flush=True)
            path = output / variant / (split + ".jsonl")
            atomic_write(path, lambda f: f.write("".join(json.dumps(r) + "\n" for r in rows).encode()))
            reports[variant][split] = dict(records=len(rows), blocks=blocks, hours=blocks * .4 / 3600, index_sha256=file_hash(path))
    report = dict(status="complete", fingerprint=fingerprint, variants=reports, diagnostics=diagnostics,
                  references=dict(count=len(refs), short=short_references))
    write_json(output / "validation.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-cache", default="data/seamless_facial_codec")
    p.add_argument("--source-manifest", help="A previously exported sources.json; rebuild without any old token caches")
    p.add_argument("--reuse-common-from", help="Validated dataset with identical inputs; reuse audio/text/reference payloads during --stage index or all")
    p.add_argument("--raw-root", default="/home/Seamless_Interaction_Dataset_ARKit/data_arkit")
    p.add_argument("--codec-root", default="../CosyVoice_Face/face_tokenizer/runs/experiments/base")
    p.add_argument("--model", required=True, help="Complete local Qwen 1.7B Base snapshot")
    p.add_argument("--output", required=True)
    p.add_argument("--codecs", nargs="+", choices=VARIANTS, default=["semantic_gan", "vae_vq"])
    p.add_argument("--stage", choices=["all", "index", "references", "encode", "validate"], default="all")
    p.add_argument("--rebuild", choices=["all", "resume"], default="resume")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--cpu-workers", type=int, default=4)
    p.add_argument("--audio-batch-size", type=int, default=4)
    p.add_argument("--max-recordings", type=int, default=0, help="Diagnostic encode limit; cannot produce a complete validation report")
    args = p.parse_args()
    if not 0 <= args.worker < args.workers:
        p.error("Require 0 <= worker < workers")
    if min(args.audio_batch_size, args.cpu_workers, args.cpu_threads) < 1:
        p.error("Batch size and CPU worker/thread counts must be positive")
    if args.validate_only:
        args.stage = "validate"
    if args.reuse_common_from and args.stage not in ("all", "index"):
        p.error("--reuse-common-from requires --stage index or all")
    contract = initialize(args)
    records = read_json(Path(args.output) / "sources.json")
    if digest(records) != contract["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    if args.reuse_common_from:
        reuse_common(args, contract, records)
    if args.stage in ("all", "references") and not args.reuse_common_from:
        prepare_references(args, contract, records)
    if args.stage in ("all", "encode"):
        encode_records(args, contract, records)
    if args.stage in ("all", "validate"):
        validate_build(args, contract, records)


if __name__ == "__main__":
    main()
