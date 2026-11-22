"""Train the Qwen Talker and facial heads, saving a resumable checkpoint every epoch."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed

from qwen_tts import Qwen3TTSModel
from .dataset import StreamDataset, collate_streams
from .files import atomic_write, digest, file_hash, model_identity, read_contract, read_json, write_json
from .model import HoldListenModel
from .protocol import Geometry, WindowPolicy


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def arguments():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    defaults = read_json(known.config) if known.config else {}
    p = argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument("--model", default=".runtime/models/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--data", default="data/qwen_holdlistenstream_v1")
    p.add_argument("--codec", default="semantic_gan")
    p.add_argument("--output", default="runs/holdlistenstream_semantic_gan")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--seed", type=int, default=1986)
    p.add_argument("--window", type=float, default=60)
    p.add_argument("--drop", type=float, default=30)
    p.add_argument("--max-length", type=int, default=4000)
    p.add_argument("--language", default="english")
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", help="Checkpoint .pt, including the latest.pt symlink")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.set_defaults(**defaults)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation) < 1:
        p.error("epochs, batch-size and accumulation must be positive")
    return args


def main():
    args = arguments()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation, mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=args.seed),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    torch.set_num_threads(4)
    set_seed(args.seed, device_specific=True)
    data_root, output = Path(args.data), Path(args.output)
    contract = read_contract(data_root / "contract.json")
    if model_identity(args.model) != contract["model_files"]:
        raise ValueError("Qwen model/tokenizer snapshot differs from the prepared data")
    asset = contract["codecs"][args.codec]
    if file_hash(asset["path"]) != asset["sha256"]:
        raise ValueError("Frozen facial codec weights changed")
    geometry = Geometry(**asset["geometry"])
    policy = WindowPolicy.from_seconds(args.window, args.drop, args.max_length)
    qwen = Qwen3TTSModel.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    qwen.model.speaker_encoder.requires_grad_(False)
    qwen.model.speech_tokenizer.model.requires_grad_(False)
    model = HoldListenModel(qwen, geometry, language=args.language, policy=policy)
    if not args.no_gradient_checkpointing:
        model.enable_checkpointing()
    train = StreamDataset(data_root, args.codec, "train", policy, geometry, model.prompt_length, args.train_limit)
    dev = StreamDataset(data_root, args.codec, "dev", policy, geometry, model.prompt_length, args.dev_limit)
    if not train or not dev:
        raise ValueError("Both training and validation require at least one window")
    settings = model.settings()
    data_fingerprint = digest(dict(contract=contract["fingerprint"], train=train.index_sha256, dev=dev.index_sha256))
    signature = digest(dict(data=data_fingerprint, settings=settings, lr=args.lr,
                            weight_decay=args.weight_decay, batch=args.batch_size, accumulation=args.accumulation,
                            seed=args.seed, world_size=accelerator.num_processes,
                            train_limit=args.train_limit, dev_limit=args.dev_limit))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    start_epoch, global_step, checkpoint = 0, 0, None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "qwen-holdlistenstream-v1" or checkpoint["signature"] != signature:
            raise ValueError("Resume model/data/training configuration mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
    elif (output / "latest.pt").exists():
        raise ValueError("Output already contains training checkpoints; use --resume or a new output")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_streams, persistent_workers=args.workers > 0)
    # Validation uses disjoint rank subsets, without DistributedSampler padding.
    # Run the unwrapped model below so unequal subset lengths need no DDP join.
    dev_subset = Subset(dev, list(range(accelerator.process_index, len(dev), accelerator.num_processes)))
    dev_loader = DataLoader(dev_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_streams, persistent_workers=args.workers > 0)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    if checkpoint is not None:
        restore_rng(checkpoint["rng"][accelerator.process_index])
        del checkpoint
        if accelerator.is_main_process:
            print(json.dumps(dict(event="resumed", checkpoint=args.resume, epoch=start_epoch,
                                  global_step=global_step, optimizer=True, scheduler=True,
                                  rng_ranks=accelerator.num_processes)), flush=True)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run.json", dict(args=vars(args), signature=signature, settings=settings,
                                             train_windows=len(train), dev_windows=len(dev), data_fingerprint=data_fingerprint))
    accelerator.wait_for_everyone()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loader.set_epoch(epoch)
        started = time.monotonic()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = model(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError("Nonfinite training loss")
                accelerator.backward(result["loss"])
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    global_step += 1
            if step % 10 == 0 and accelerator.is_main_process:
                row = dict(event="train", epoch=epoch + 1, batch=step, global_step=global_step,
                           seconds=round(time.monotonic() - started, 3),
                           **{key: float(result[key]) for key in ("loss", "audio_loss", "audio_aux_loss", "face_loss")})
                print(json.dumps(row), flush=True)
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
        model.eval()
        totals = torch.zeros(5, device=accelerator.device, dtype=torch.float64)
        with torch.inference_mode(), accelerator.autocast():
            for batch in dev_loader:
                result = accelerator.unwrap_model(model)(batch)
                n = result["supervised_blocks"]
                totals[:4] += torch.stack([result[key].double() for key in ("loss", "audio_loss", "audio_aux_loss", "face_loss")]) * n
                totals[4] += n
        totals = accelerator.reduce(totals, reduction="sum")
        if not torch.isfinite(totals).all() or totals[4] == 0:
            raise FloatingPointError("Invalid validation loss")
        rng = gather_object([rng_state()])
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            payload = dict(format="qwen-holdlistenstream-v1", signature=signature, settings=settings,
                           model=unwrapped.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           rng=rng, epoch=epoch + 1, global_step=global_step, codec=asset,
                           base_model=str(Path(args.model).resolve()), data_fingerprint=data_fingerprint,
                           base_identity=contract["model_files"],
                           validation=(totals[:4] / totals[4]).tolist())
            destination = output / f"checkpoint-epoch-{epoch + 1:04d}.pt"
            atomic_write(destination, lambda f: torch.save(payload, f))
            temporary = output / "latest.pt.tmp"
            if temporary.is_symlink():
                temporary.unlink()
            temporary.symlink_to(destination.name)
            os.replace(temporary, output / "latest.pt")
            row = dict(event="epoch_saved", epoch=epoch + 1, global_step=global_step,
                       validation=payload["validation"], seconds=round(time.monotonic() - started, 3), checkpoint=str(destination))
            print(json.dumps(row), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
