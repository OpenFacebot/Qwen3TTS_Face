"""Compare cached generation and run a bounded three-minute streaming session."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from qwen_tts.holdlistenstream.files import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=".runtime/models/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", required=True, help="Prepared reference .npz")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ticks", type=int, default=450)
    args = p.parse_args()
    if args.ticks < 1:
        p.error("ticks must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Compare greedy paths in FP32: BF16 kernel rounding can legitimately break
    # near-tied argmax decisions and then diverge autoregressively.
    qwen = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, torch_dtype=torch.float32,
                                        attn_implementation="sdpa")
    with np.load(args.reference, allow_pickle=False) as f:
        speaker = f["speaker"].copy()
    cached = qwen.start_holdlistenstream(args.checkpoint, speaker_embedding=speaker)
    full = qwen.start_holdlistenstream(args.checkpoint, speaker_embedding=speaker, cache=False)
    parity = []
    for text, speaking in [("Hello, how are you?", True), ("", True), ("", False), ("I am listening.", True)]:
        a, b = cached.tick(text, speaking), full.tick(text, speaking)
        parity.append(dict(audio_equal=bool(np.array_equal(a.audio_codes, b.audio_codes)),
                           face_equal=bool(np.array_equal(a.face_codes, b.face_codes))))
    cached.close()
    full.close()
    if not all(r["audio_equal"] and r["face_equal"] for r in parity):
        write_json(output / "parity.json", parity)
        raise AssertionError("Cached and full-prefix greedy generation differ; see parity.json")
    del cached, full
    qwen.model.to(dtype=torch.bfloat16)
    qwen.model.speech_tokenizer.model.to(dtype=torch.bfloat16)
    stream = qwen.start_holdlistenstream(args.checkpoint, speaker_embedding=speaker)
    audio, motion, audio_codes, face_codes, elapsed, positions = [], [], [], [], [], []
    rollovers, previous_blocks, max_face_history = 0, 0, 0
    if stream.model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(stream.model.device)
    start = time.monotonic()
    for tick in range(args.ticks):
        speaking = tick % 50 < 30
        text = "Hello, this is a streaming voice and facial motion test." if tick % 50 == 0 else ""
        chunk = stream.tick(text, speaking)
        audio.append(chunk.audio)
        motion.append(chunk.motion)
        audio_codes.append(chunk.audio_codes)
        face_codes.append(chunk.face_codes)
        elapsed.append(chunk.elapsed_seconds)
        positions.append(stream.position)
        assert len(stream.blocks) <= stream.model.policy.window_blocks
        assert stream.position <= stream.model.policy.max_length
        assert stream.past.get_seq_length() == stream.position
        assert stream.audio_history.shape[-1] <= 25
        face_history = stream.face_state.history.shape[1]
        assert face_history <= stream.face.decoder_blocks - 1
        max_face_history = max(max_face_history, face_history)
        rollovers += int(len(stream.blocks) < previous_blocks)
        previous_blocks = len(stream.blocks)
        if tick % 25 == 0:
            print(json.dumps(dict(tick=tick, seconds=round(time.monotonic() - start, 2), lm_positions=stream.position)), flush=True)
    sf.write(output / "audio.wav", np.concatenate(audio), 24000)
    np.save(output / "motion.npy", np.concatenate(motion), allow_pickle=False)
    np.savez_compressed(output / "tokens.npz", audio=np.concatenate(audio_codes), face=np.concatenate(face_codes))
    report = dict(ticks=args.ticks, seconds_of_media=args.ticks * .4, audio_samples=sum(len(x) for x in audio),
                  motion_frames=sum(len(x) for x in motion), max_lm_positions=max(positions),
                  first_tick_seconds=elapsed[0], mean_tick_seconds=float(np.mean(elapsed)),
                  p95_tick_seconds=float(np.percentile(elapsed, 95)), real_time_factor=sum(elapsed) / (args.ticks * .4),
                  window_rollovers=rollovers, max_face_history_groups=max_face_history,
                  peak_cuda_memory_bytes=(torch.cuda.max_memory_allocated(stream.model.device)
                                          if stream.model.device.type == "cuda" else None),
                  parity=parity, parity_dtype="float32", streaming_dtype="bfloat16")
    stream.close()
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
