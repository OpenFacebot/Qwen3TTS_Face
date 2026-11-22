"""One clock and window policy for data preparation, training and inference.

Turn control and rolling text history follow CosyVoice_Face's holdlistenstream.
Audio occupies five (16-codebook) frames per tick, rather than ten Cosy frames.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Integral

HOLD, LISTEN = -1, -2
TEXT_SLOTS, AUDIO_FRAMES, FACE_FRAMES, SAMPLE_RATE = 4, 5, 10, 24000
TICK_SAMPLES = 9600
VARIANTS = ("vae_vq", "vae_fsq", "semantic", "semantic_first", "semantic_gan", "semantic_first_gan")


class TurnSlots:
    def __init__(self):
        self.pending = deque()
        self.speaking = False

    def tick(self, ids=(), *, speaking=None):
        ids = list(ids)
        if any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in ids):
            raise ValueError("Text arrivals must be nonnegative integer token IDs")
        if speaking is not None and type(speaking) is not bool:
            raise ValueError("speaking must be a boolean or None")
        if speaking is False and ids:
            raise ValueError("A listening tick cannot also deliver speech text")
        if speaking is not None:
            self.speaking = speaking
        elif ids:
            self.speaking = True
        if not self.speaking:
            self.pending.clear()
            return [LISTEN] * TEXT_SLOTS
        self.pending.extend(ids)
        return [self.pending.popleft() if self.pending else HOLD for _ in range(TEXT_SLOTS)]


def block_at(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid event time")
    return math.ceil(seconds / .4 - 1e-8)


def annotation_slots(annotation, duration_frames, encode, release_delay=.8):
    """Deliver complete transcript segments at their start/available_at timestamp."""
    if duration_frames < 1 or not math.isfinite(release_delay) or release_delay < 0:
        raise ValueError("Invalid duration or release delay")
    count = math.ceil(duration_frames / FACE_FRAMES)
    activity = [0] * (count + 1)
    arrivals = defaultdict(list)
    events = []

    def mark(start, end):
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (start, end)) or end <= start:
            raise ValueError("Invalid annotated speech interval")
        activity[min(count, block_at(max(0, start)))] += 1
        activity[min(count, block_at(max(0, end + release_delay)))] -= 1

    for interval in annotation["metadata:vad"]:
        mark(interval["start"], interval["end"])
    for segment in annotation["metadata:transcript"]:
        words = segment.get("words") or []
        text = (segment.get("transcript") or "").strip() or " ".join((w.get("word") or "").strip() for w in words).strip()
        if not text:
            continue
        start, end = segment.get("start"), segment.get("end")
        if start is None or end is None:
            timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
            if not timed:
                raise ValueError("Transcript has no speech interval")
            start = min(w["start"] for w in timed) if start is None else start
            end = max(w["end"] for w in timed) if end is None else end
        mark(start, end)
        at = segment.get("available_at", start)
        if not math.isfinite(at):
            raise ValueError("Invalid transcript arrival")
        events.append((max(0, at), text))
    for at, text in sorted(events, key=lambda event: event[0]):
        arrivals[block_at(at)].extend(encode(text))
    clock, active, result = TurnSlots(), 0, []
    for b in range(count):
        active += activity[b]
        ids = arrivals.get(b, ())
        result.append(clock.tick(ids, speaking=bool(active or ids or clock.pending)))
    return result


@dataclass(frozen=True)
class Geometry:
    variant: str
    sizes: tuple
    hop: int = 5
    fps: int = 25

    def __post_init__(self):
        if self.variant not in VARIANTS or self.hop != 5 or self.fps != 25:
            raise ValueError("This stream requires a supported 25 Hz / hop-5 codec")
        expected = 1 if self.variant.startswith("vae_") else 5
        if len(self.sizes) != expected or any(s < 2 for s in self.sizes):
            raise ValueError("Unexpected codec codebook layout")

    @property
    def mtp(self):
        return self.variant in ("semantic", "semantic_gan")

    @property
    def primary_books(self):
        return 2 if self.mtp else len(self.sizes)

    @property
    def positions(self):
        return TEXT_SLOTS + AUDIO_FRAMES + 2 * self.primary_books


@dataclass(frozen=True)
class WindowPolicy:
    window_blocks: int = 150
    drop_blocks: int = 75
    max_length: int = 4000

    def __post_init__(self):
        if not 0 < self.drop_blocks < self.window_blocks or self.max_length < 1:
            raise ValueError("Require 0 < drop < window and a positive sequence budget")

    @classmethod
    def from_seconds(cls, window=60, drop=30, max_length=4000):
        values = [Decimal(str(x)) / Decimal(".4") for x in (window, drop)]
        if any(not x.is_finite() or x != x.to_integral_value() for x in values):
            raise ValueError("Window and drop must be multiples of 0.4 seconds")
        return cls(*(int(x) for x in values), max_length)

    def prefix_limit(self, geometry, prompt_length):
        limit = self.max_length - prompt_length - 2 - self.window_blocks * geometry.positions
        if limit < 0:
            raise ValueError("Full window exceeds sequence budget")
        return limit

    def spans(self, count):
        """(left, right, loss_from_block), including the final partial window."""
        end = min(count, self.window_blocks)
        if end:
            yield 0, end, 0
        while end < count:
            start = end - (self.window_blocks - self.drop_blocks)
            new_end = min(count, end + self.drop_blocks)
            yield start, new_end, end - start
            end = new_end


def roll_text_history(previous, evicted, first_retained, limit):
    text = list(previous)
    for slots in evicted:
        if int(slots[0]) == LISTEN:
            text.clear()
        else:
            text.extend(int(x) for x in slots if x >= 0)
    if int(first_retained[0]) == LISTEN:
        text.clear()
    return text[-limit:] if limit else []


def validate_slots(slots, vocab_size):
    import numpy as np
    x = np.asarray(slots)
    if x.ndim != 2 or x.shape[1] != TEXT_SLOTS or x.dtype.kind not in "iu":
        raise ValueError("Expected integer [blocks,4] text slots")
    if ((x < LISTEN) | (x >= vocab_size)).any():
        raise ValueError("Text slot outside vocabulary")
    listen = x == LISTEN
    if (listen.any(1) != listen.all(1)).any() or ((x[:, :-1] == HOLD) & (x[:, 1:] >= 0)).any():
        raise ValueError("Expected LISTEN or text followed by HOLD")
