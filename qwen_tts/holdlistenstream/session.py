"""Bounded streaming inference with independent LM, audio and motion histories."""
from collections import deque
from dataclasses import dataclass
import time
import numpy as np
import torch

from .codecs import load_face
from .files import file_hash, model_identity, digest
from .model import adapter_from_settings
from .protocol import TurnSlots, roll_text_history, TICK_SAMPLES


@dataclass
class StreamChunk:
    audio: np.ndarray
    motion: np.ndarray
    audio_codes: np.ndarray
    face_codes: np.ndarray
    text_slots: list
    sample_rate: int = 24000
    motion_fps: int = 25
    elapsed_seconds: float = 0


class HoldListenSession:
    def __init__(self, qwen, checkpoint, *, ref_audio=None, speaker_embedding=None,
                 codec_checkpoint=None, do_sample=False, temperature=.8, top_k=50,
                 cache=True):
        if (ref_audio is None) == (speaker_embedding is None):
            raise ValueError("Provide exactly one of ref_audio and speaker_embedding")
        # Training checkpoints also contain two optimizer moments per weight.
        # Map them lazily so opening a stream reads only the model tensors.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if payload.get("format") != "qwen-holdlistenstream-v1":
            raise ValueError("Not a holdlistenstream checkpoint")
        expected_base = payload.get("base_identity")
        if expected_base is None:
            raise ValueError("Checkpoint is missing its base-model/tokenizer identity")
        if getattr(qwen, "_holdlisten_base_fingerprint", None) != digest(expected_base):
            from pathlib import Path
            from transformers.utils.hub import cached_file
            config_path = cached_file(qwen.model.config._name_or_path, "config.json", local_files_only=True)
            if model_identity(Path(config_path).parent) != expected_base:
                raise ValueError("Inference base model/tokenizer differs from the training snapshot")
            qwen._holdlisten_base_fingerprint = digest(expected_base)
        self.model = adapter_from_settings(qwen, payload["settings"])
        self.model.load_state_dict(payload["model"], strict=True)
        device, dtype = qwen.model.talker.device, qwen.model.talker.dtype
        self.model.to(device=device, dtype=dtype).eval()
        self.codec_asset = payload["codec"]
        face_path = codec_checkpoint or self.codec_asset["path"]
        if file_hash(face_path) != self.codec_asset["sha256"]:
            raise ValueError("Facial codec does not match the training checkpoint")
        self.face, geometry, _ = load_face(face_path, self.model.geometry.variant, device)
        if geometry.sizes != tuple(self.model.geometry.sizes):
            raise ValueError("Facial codebook mismatch")
        self.audio_model = qwen.model.speech_tokenizer.model.to(device).eval()
        if speaker_embedding is None:
            if isinstance(ref_audio, str):
                waveform, sr = qwen._load_audio_to_np(ref_audio)
            elif isinstance(ref_audio, tuple) and len(ref_audio) == 2:
                waveform, sr = np.asarray(ref_audio[0], dtype=np.float32), int(ref_audio[1])
            else:
                raise TypeError("Reference audio must be a path or (waveform, sample_rate)")
            if waveform.ndim == 2:
                waveform = waveform.mean(-1)
            if waveform.ndim != 1 or not np.isfinite(waveform).all() or sr <= 0:
                raise ValueError("Invalid reference waveform")
            if sr != 24000:
                from scipy.signal import resample_poly
                from math import gcd
                divisor = gcd(sr, 24000)
                waveform = resample_poly(waveform, 24000 // divisor, sr // divisor).astype(np.float32)
            if len(waveform) < 1024:
                raise ValueError("Reference audio is too short for the speaker encoder")
            qwen.model.speaker_encoder.to(device)
            with torch.inference_mode():
                speaker_embedding = qwen.model.extract_speaker_embedding(audio=waveform, sr=24000)
        self.speaker = torch.as_tensor(speaker_embedding, device=device, dtype=dtype).reshape(-1)
        self.tokenizer = qwen.processor.tokenizer
        self.clock, self.blocks, self.history = TurnSlots(), deque(), []
        self.face_state = None
        self.audio_history = torch.empty((1, 16, 0), device=device, dtype=torch.long)
        self.cache_enabled, self.past, self.hidden = cache, None, None
        self.full_embeddings, self.position, self.closed, self.total_blocks = [], 0, False, 0
        self.do_sample, self.temperature, self.top_k = do_sample, temperature, top_k
        if temperature <= 0 or top_k < 0:
            raise ValueError("temperature must be positive and top_k nonnegative")
        self._rebuild()

    @torch.inference_mode()
    def _feed(self, embeddings):
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        if self.cache_enabled:
            length = embeddings.shape[1]
            positions = torch.arange(self.position, self.position + length, device=self.model.device)
            out = self.model.talker.model(inputs_embeds=embeddings,
                    attention_mask=torch.ones((1, self.position + length), dtype=torch.long, device=self.model.device),
                    position_ids=positions[None, None].expand(3, 1, -1), cache_position=positions,
                    past_key_values=self.past, use_cache=True, return_dict=True)
            self.past = out.past_key_values
            self.position += length
        else:
            self.full_embeddings.append(embeddings)
            inputs = torch.cat(self.full_embeddings, 1)
            self.position = inputs.shape[1]
            positions = torch.arange(self.position, device=self.model.device)
            out = self.model.talker.model(inputs_embeds=inputs,
                    attention_mask=torch.ones((1, self.position), dtype=torch.long, device=self.model.device),
                    position_ids=positions[None, None].expand(3, 1, -1), use_cache=False, return_dict=True)
        self.hidden = out.last_hidden_state[:, -1]

    @torch.inference_mode()
    def _rebuild(self):
        self.past, self.position, self.full_embeddings = None, 0, []
        inputs = [self.model.prompt(self.speaker, self.history)]
        if self.blocks:
            inputs.append(self.model.block_embed([b[0] for b in self.blocks],
                         np.concatenate([b[1] for b in self.blocks]), np.concatenate([b[2] for b in self.blocks])))
        self._feed(torch.cat(inputs, 0))

    def _sample(self, logits):
        if not self.do_sample:
            return logits.argmax(-1)
        logits = logits.float() / self.temperature
        if self.top_k:
            threshold = logits.topk(min(self.top_k, logits.shape[-1]), -1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)

    @torch.inference_mode()
    def tick(self, text_delta="", speaking=None):
        if self.closed:
            raise RuntimeError("Session is closed")
        if not isinstance(text_delta, str) or (speaking is not None and type(speaking) is not bool):
            raise TypeError("tick expects text_delta: str and speaking: bool or None")
        start = time.perf_counter()
        policy = self.model.policy
        if len(self.blocks) == policy.window_blocks:
            evicted = [self.blocks.popleft() for _ in range(policy.drop_blocks)]
            limit = policy.prefix_limit(self.model.geometry, self.model.prompt_length)
            self.history = roll_text_history(self.history, [x[0] for x in evicted], self.blocks[0][0], limit)
            self._rebuild()
        slots = self.clock.tick(self.tokenizer.encode(text_delta, add_special_tokens=False), speaking=speaking)
        self._feed(self.model.slot_embed(slots))
        audio_codes = []
        for _ in range(5):
            codes = self.model.predict_audio(self.hidden, self._sample)
            if codes.shape != (1, 16):
                raise RuntimeError("Audio predictor did not produce 16 codebooks")
            audio_codes.append(codes)
            self._feed(self.model.audio_embed(codes))
        face_codes = []
        for _ in range(2):
            before = self.hidden
            primary = []
            for book, head in enumerate(self.model.face_heads):
                code = self._sample(head(self.hidden))
                primary.append(code)
                if self.model.geometry.mtp and book == 1:
                    codes = self.model.face_mtp.generate(before, torch.stack(primary, -1), self._sample)
                    self._feed(self.model.face_embed(codes)[:, 1])
                else:
                    self._feed(self.model.face_embeddings[book](code))
            if not self.model.geometry.mtp:
                codes = torch.stack(primary, -1)
            face_codes.append(codes)
        ac = torch.cat(audio_codes, 0)
        fc = torch.cat(face_codes, 0)
        decoder = self.audio_model.decoder
        history = torch.cat((self.audio_history, ac.T[None]), -1)
        audio = decoder(history)[0, 0, -TICK_SAMPLES:].float().cpu().numpy()
        self.audio_history = history[..., -25:].clone()
        from facial_codec.tokens import TokenBatch
        tokens = TokenBatch(fc[None], torch.tensor([10], device=fc.device), self.face.metadata())
        motion, self.face_state = self.face.decode_stream(tokens, self.face_state)
        motion = motion[0].float().cpu().numpy()
        ac_np, fc_np = ac.cpu().numpy(), fc.cpu().numpy()
        if audio.shape != (TICK_SAMPLES,) or motion.shape != (10, 61) or not np.isfinite(audio).all() or not np.isfinite(motion).all():
            raise RuntimeError("Invalid decoded media chunk")
        self.blocks.append((slots, ac_np, fc_np))
        self.total_blocks += 1
        return StreamChunk(audio, motion, ac_np, fc_np, slots, elapsed_seconds=time.perf_counter() - start)

    def close(self):
        """Close without generating unrequested ticks or silently draining queued text."""
        self.closed = True
        self.clock.pending.clear()
        self.blocks.clear()
        self.history.clear()
        self.past, self.hidden, self.face_state = None, None, None
        self.full_embeddings.clear()
        self.audio_history = self.audio_history[..., :0]
