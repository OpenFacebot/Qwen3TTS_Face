"""Optional integration checks against the real, frozen local codec assets."""
import io
import os
from pathlib import Path
import zipfile
import numpy as np
import pytest
import torch

from qwen_tts.holdlistenstream.codecs import encode_face, load_face
from qwen_tts.holdlistenstream.files import read_json
from qwen_tts.holdlistenstream.motion import prepare_motion
from qwen_tts.holdlistenstream.protocol import VARIANTS

ROOT = os.environ.get("HOLDLISTEN_CODEC_ROOT")
MANIFEST = os.environ.get("HOLDLISTEN_SOURCE_MANIFEST")
DEVICE = os.environ.get("HOLDLISTEN_TEST_DEVICE", "cpu")
pytestmark = pytest.mark.skipif(not ROOT or not MANIFEST, reason="Set HOLDLISTEN_CODEC_ROOT and HOLDLISTEN_SOURCE_MANIFEST for asset tests")


def raw_sample():
    record = read_json(MANIFEST)[0]
    with zipfile.ZipFile(record["archive"]) as archive:
        raw = np.load(io.BytesIO(archive.read(record["person"] + "_arkit61.npy")), allow_pickle=False)
    return raw


@pytest.mark.parametrize("variant", VARIANTS)
@torch.inference_mode()
def test_real_codec_chunk_boundaries_and_stream_decode(variant):
    torch.set_num_threads(4)
    model, geometry, asset = load_face(Path(ROOT) / variant / "checkpoints/epoch_005.pt", variant, DEVICE)
    motion, _ = prepare_motion(raw_sample())
    motion = motion[:min(1200, len(motion) // 10 * 10)]
    full = model.encode(torch.from_numpy(motion)[None].to(DEVICE))
    chunked = encode_face(model, motion, chunk_frames=100)
    np.testing.assert_array_equal(chunked, full.codes[0].cpu().numpy())
    output = model.decode(full)
    from facial_codec.tokens import TokenBatch
    state, pieces = None, []
    for start in range(0, full.codes.shape[1], 2):
        codes = full.codes[:, start:start + 2]
        tokens = TokenBatch(codes, torch.tensor([codes.shape[1] * 5], device=DEVICE), asset["metadata"])
        part, state = model.decode_stream(tokens, state)
        pieces.append(part)
        assert state.history.shape[1] <= model.decoder_blocks - 1
    torch.testing.assert_close(torch.cat(pieces, 1), output, atol=2e-5, rtol=2e-5)


def test_preprocessing_matches_original_v2():
    legacy = os.environ.get("HOLDLISTEN_LEGACY_TOOLS")
    if not legacy:
        pytest.skip("Set HOLDLISTEN_LEGACY_TOOLS for the original V2 implementation")
    import sys
    from argparse import Namespace
    sys.path.insert(0, legacy)
    try:
        from v2_preprocessing.motion_sampling import resample_interp1d
        from v2_preprocessing.motion_filter import fill_nonfinite, filter_values
        raw = raw_sample()
        old = resample_interp1d(raw.astype(np.float64), 30., 25., kind="cubic")
        old, _ = fill_nonfinite(old)
        old = filter_values(old, 25., Namespace(face_hz=7., head_hz=3., eye_hz=2., order=4, roll_median=5, no_clamp=False)).astype(np.float32)
        new, _ = prepare_motion(raw)
        np.testing.assert_array_equal(new, old)
    finally:
        sys.path.remove(legacy)
