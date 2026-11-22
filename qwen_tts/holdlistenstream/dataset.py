"""Read independently verifiable caches and sample rolling training windows."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from .files import read_contract, read_json, file_hash
from .protocol import roll_text_history


class StreamDataset(Dataset):
    def __init__(self, root, variant, split, policy, geometry, prompt_length, limit=0):
        self.root = Path(root)
        self.contract = read_contract(self.root / "contract.json")
        report = read_json(self.root / "validation.json")
        index = self.root / variant / (split + ".jsonl")
        if report["status"] != "complete" or report["fingerprint"] != self.contract["fingerprint"]:
            raise ValueError("Dataset build has not passed full validation")
        self.index_sha256 = file_hash(index)
        if self.index_sha256 != report["variants"][variant][split]["index_sha256"]:
            raise ValueError("Dataset index checksum mismatch")
        self.records = [json.loads(line) for line in index.read_text().splitlines() if line]
        self.prefix_limit = policy.prefix_limit(geometry, prompt_length)
        self.windows = [(i, start, end, loss) for i, r in enumerate(self.records) for start, end, loss in policy.spans(r["blocks"])]
        if limit:
            # Deterministic spread across the corpus, including short and long turns.
            indices = np.random.default_rng(1986).choice(len(self.windows), min(limit, len(self.windows)), replace=False)
            self.windows = [self.windows[int(i)] for i in indices]

    def __len__(self):
        return len(self.windows)

    @lru_cache(maxsize=4)
    def load(self, index):
        r = self.records[index]
        for key in ("common", "face", "reference"):
            if file_hash(self.root / r[key]) != r["checksums"][key]:
                raise ValueError(f"Training cache checksum mismatch: {r[key]}")
        with np.load(self.root / r["common"], allow_pickle=False) as c, np.load(self.root / r["face"], allow_pickle=False) as f, np.load(self.root / r["reference"], allow_pickle=False) as s:
            return dict(text=c["text"].copy(), audio=c["audio"].copy(), face=f["face"].copy(), speaker=s["speaker"].copy())

    def __getitem__(self, index):
        record, start, end, loss = self.windows[index]
        data = self.load(record)
        history = roll_text_history([], data["text"][:start], data["text"][start], self.prefix_limit)
        return dict(utt=self.records[record]["utt"], text=data["text"][start:end], audio=data["audio"][start * 5:end * 5],
                    face=data["face"][start * 2:end * 2], speaker=data["speaker"], history=history, loss_from_block=loss)


def collate_streams(samples):
    return samples
