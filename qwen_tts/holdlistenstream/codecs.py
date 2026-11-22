"""Frozen codec assets and bounded encoders; no teacher dependencies."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from .files import file_hash
from .protocol import Geometry


def load_face(path, variant, device="cpu"):
    from facial_codec.training.checkpoint import load_model
    # TF32 can move nearest-codebook decisions across boundaries. Match the
    # original full-precision extraction in both preparation and live decoding.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, checkpoint = load_model(path, device)
    model.requires_grad_(False)
    geometry = Geometry(variant, tuple(model.sizes), model.hop, model.config["data"]["fps"])
    if model.data_dim != 61:
        raise ValueError("holdlistenstream requires the canonical 61 motion channels")
    actual = checkpoint["config"]["model"]["type"]
    expected = "vae" if variant.startswith("vae_") else variant.removesuffix("_gan")
    if actual != expected:
        raise ValueError(f"{variant} cannot use {actual} checkpoint")
    if variant.startswith("vae_") and checkpoint["config"]["model"]["quantizer"] != variant.removeprefix("vae_"):
        raise ValueError("VAE quantizer does not match the requested variant")
    if bool(checkpoint.get("discriminator") is not None) != variant.endswith("_gan"):
        raise ValueError("GAN checkpoint does not match the requested variant")
    return model, geometry, {"path": str(Path(path).resolve()), "sha256": file_hash(path),
                             "metadata": model.metadata(), "geometry": asdict(geometry)}


@torch.inference_mode()
def encode_face(model, motion, chunk_frames=750):
    """Encode complete hop-aligned chunks with the finite encoder left context."""
    if chunk_frames % model.hop:
        raise ValueError("Motion chunks must end on a codec hop")
    device = next(model.parameters()).device
    history = (model.encoder_blocks - 1) * model.hop
    output = []
    for start in range(0, len(motion), chunk_frames):
        left, end = max(0, start - history), min(len(motion), start + chunk_frames)
        chunk = torch.from_numpy(motion[left:end].copy()).unsqueeze(0).to(device)
        codes = model.encode(chunk).codes[0, (start - left) // model.hop:]
        output.append(codes.cpu().numpy().astype(np.uint16))
    return np.concatenate(output)


@torch.inference_mode()
def encode_audio(tokenizer, waveform, chunk_seconds=30):
    """Mimi carries convolution padding and a bounded attention cache between chunks."""
    return encode_audio_batch(tokenizer, [waveform], chunk_seconds)[0]


@torch.inference_mode()
def encode_audio_batch(tokenizer, waveforms, chunk_seconds=30):
    """Independent continuous streams, padded only to the longest batch member.

    The encoder is causal: padding after a recording cannot change its valid
    codes. Keeping equal-duration recordings together avoids wasted tail work.
    """
    if not waveforms or any(len(w) == 0 for w in waveforms):
        raise ValueError("Audio batch must contain nonempty waveforms")
    encoder = tokenizer.model.encoder
    encoder.config.use_cache = True
    encoder.encoder_transformer.config.use_cache = True
    size = int(chunk_seconds * 24000)
    if size % 1920:
        raise ValueError("Audio chunks must end on a 1920-sample boundary")
    past = padding = None
    codes = []
    device, dtype = next(encoder.parameters()).device, next(encoder.parameters()).dtype
    maximum = max(len(w) for w in waveforms)
    for start in range(0, maximum, size):
        padded = np.zeros((len(waveforms), 1, min(size, maximum - start)), dtype=np.float32)
        for index, waveform in enumerate(waveforms):
            part = waveform[start:start + size]
            padded[index, 0, :len(part)] = part
        audio = torch.from_numpy(padded).to(device=device, dtype=dtype)
        encoded = encoder.encode(audio, num_quantizers=16, encoder_past_key_values=past,
                                 padding_cache=padding, use_streaming=True, return_dict=True)
        past, padding = encoded.encoder_past_key_values, encoded.padding_cache
        codes.append(encoded.audio_codes[:, :16].transpose(1, 2).cpu().numpy().astype(np.uint16))
    joined = np.concatenate(codes, axis=1)
    return [joined[i, :(len(w) + 1919) // 1920] for i, w in enumerate(waveforms)]


def load_speaker(model_path, device):
    """Load only the frozen speaker encoder from a trusted local model snapshot."""
    from safetensors import safe_open
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
    config = Qwen3TTSConfig.from_pretrained(model_path)
    model = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
    state = {}
    for path in Path(model_path).glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("speaker_encoder."):
                    state[key.removeprefix("speaker_encoder.")] = f.get_tensor(key)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def speaker_vector(model, waveform):
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    x = torch.from_numpy(waveform.copy()).unsqueeze(0)
    mel = mel_spectrogram(x, n_fft=1024, num_mels=128, sampling_rate=24000,
                          hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
    result = model(mel.to(next(model.parameters()).device))[0].float().cpu().numpy()
    if not np.isfinite(result).all() or result.shape != (2048,):
        raise ValueError("Invalid 1.7B speaker vector")
    return result
