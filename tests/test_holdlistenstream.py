"""Contract tests that catch temporal leaks and train/inference disagreement."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from qwen_tts.holdlistenstream.protocol import (TurnSlots, HOLD, LISTEN, Geometry, WindowPolicy,
                                               annotation_slots, roll_text_history, validate_slots, VARIANTS)
from qwen_tts.holdlistenstream.motion import prepare_motion


def test_fifo_and_listen_discards_backlog():
    c = TurnSlots()
    assert c.tick(range(6), speaking=True) == [0, 1, 2, 3]
    assert c.tick(speaking=True) == [4, 5, HOLD, HOLD]
    c.tick(range(8), speaking=True)
    assert c.tick(speaking=False) == [LISTEN] * 4
    assert c.tick(speaking=True) == [HOLD] * 4
    with pytest.raises(ValueError, match="listening tick"):
        c.tick([7], speaking=False)
    assert c.tick(speaking=False) == [LISTEN] * 4
    assert c.tick([7]) == [7, HOLD, HOLD, HOLD]
    assert c.tick() == [HOLD] * 4
    with pytest.raises(ValueError, match="boolean"):
        c.tick(speaking="false")


def test_annotation_sentence_arrival_and_release():
    a = {"metadata:vad": [{"start": .4, "end": 1.2}],
         "metadata:transcript": [{"start": .4, "end": 1.2, "transcript": "hello", "available_at": .8}]}
    slots = annotation_slots(a, 70, lambda text: [10, 11, 12, 13, 14])
    assert slots[0] == [LISTEN] * 4
    assert slots[1] == [HOLD] * 4
    assert slots[2] == [10, 11, 12, 13]
    assert slots[3] == [14, HOLD, HOLD, HOLD]
    assert slots[4] == [HOLD] * 4
    assert slots[5] == [LISTEN] * 4
    validate_slots(np.asarray(slots), 100)


def test_rolling_supervision_covers_each_block_once():
    p = WindowPolicy()
    coverage = np.zeros(503, dtype=int)
    spans = list(p.spans(len(coverage)))
    assert spans[:2] == [(0, 150, 0), (75, 225, 75)]
    for start, end, loss in spans:
        coverage[start + loss:end] += 1
        assert end - start <= 150
    assert (coverage == 1).all()
    with pytest.raises(ValueError):
        WindowPolicy.from_seconds(60, .3)


def test_history_has_no_future_text_and_resets_at_listen():
    old = [[4, 5, -1, -1], [-2] * 4, [6, 7, 8, -1]]
    assert roll_text_history([1, 2, 3], old, [9, -1, -1, -1], 2) == [7, 8]
    assert roll_text_history([1], old, [-2] * 4, 20) == []


@pytest.mark.parametrize("variant", VARIANTS)
def test_geometry(variant):
    g = Geometry(variant, (1024,) if variant.startswith("vae_") else (1024,) * 5)
    expected = 11 if variant.startswith("vae_") else 13 if g.mtp else 19
    assert g.positions == expected
    assert WindowPolicy().prefix_limit(g, 10) >= 0


def test_slots_reject_mixed_listen_and_text_after_hold():
    for bad in ([[1, -2, -2, -2]], [[1, -1, 2, -1]], [[100, -1, -1, -1]]):
        with pytest.raises(ValueError):
            validate_slots(np.asarray(bad), 100)


def test_motion_preserves_clock_and_repairs_missing_values():
    raw = np.zeros((120, 61), dtype=np.float32)
    raw[:, :52] = .4
    raw[30] = np.nan
    motion, report = prepare_motion(raw)
    assert motion.shape == (100, 61)
    assert np.isfinite(motion).all()
    assert np.allclose(motion[:, :52], .4)
    assert report["raw_nonfinite"] == 61


def tiny_adapter(variant):
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
    from qwen_tts.holdlistenstream.model import HoldListenModel
    config = Qwen3TTSConfig(tts_pad_token_id=1, tts_bos_token_id=2, tts_eos_token_id=3,
        talker_config=dict(hidden_size=32, text_hidden_size=32, vocab_size=64, text_vocab_size=128,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, num_code_groups=16, rope_scaling=dict(rope_type="default", mrope_section=[1, 1, 2], interleaved=True),
            codec_think_id=32, codec_nothink_id=33, codec_think_bos_id=34, codec_think_eos_id=35,
            codec_pad_id=36, codec_bos_id=37, codec_language_id={"english": 38},
            code_predictor_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64, num_code_groups=16)))
    config.talker_config._attn_implementation = "sdpa"
    config.talker_config.code_predictor_config._attn_implementation = "sdpa"
    talker = Qwen3TTSTalkerForConditionalGeneration(config.talker_config)
    qwen = SimpleNamespace(model=SimpleNamespace(talker=talker, config=config),
                           processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [4, 5, 6])))
    g = Geometry(variant, (16,) if variant.startswith("vae_") else (16,) * 5)
    return HoldListenModel(qwen, g, mtp_width=32, mtp_layers=2)


def sample(g, blocks=3, loss=0):
    rng = np.random.default_rng(1)
    return dict(text=np.asarray([[7, 8, -1, -1], [-1] * 4, [-2] * 4][:blocks]),
                audio=rng.integers(0, 16, size=(blocks * 5, 16)),
                face=rng.integers(0, 16, size=(blocks * 2, len(g.sizes))),
                speaker=np.ones(32, dtype=np.float32), history=[9], loss_from_block=loss)


@pytest.mark.parametrize("variant", VARIANTS)
def test_trainable_heads_and_finite_gradients(variant):
    torch.manual_seed(5)
    model = tiny_adapter(variant)
    result = model([sample(model.geometry)])
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_causal_hidden_does_not_see_target_or_future():
    model = tiny_adapter("semantic_gan").eval()
    s = sample(model.geometry)
    x = model.block_embed(s["text"], s["audio"], s["face"])[None]
    position = 4  # First audio frame follows the four text slots.
    changed = x.clone()
    changed[:, position:] += torch.randn_like(changed[:, position:])
    with torch.no_grad():
        a = model.talker.model(inputs_embeds=x, use_cache=False).last_hidden_state
        b = model.talker.model(inputs_embeds=changed, use_cache=False).last_hidden_state
    torch.testing.assert_close(a[:, :position], b[:, :position])


def test_cached_backbone_matches_full_prefix():
    model = tiny_adapter("vae_vq").eval()
    s = sample(model.geometry)
    x = torch.cat((model.prompt(s["speaker"], []), model.block_embed(s["text"], s["audio"], s["face"])))[None]
    past, chunks = None, []
    with torch.no_grad():
        full = model.talker.model(inputs_embeds=x, use_cache=False).last_hidden_state
        for start in range(0, x.shape[1], 4):
            part = x[:, start:start + 4]
            end = start + part.shape[1]
            p = torch.arange(start, end)
            out = model.talker.model(inputs_embeds=part, past_key_values=past, use_cache=True,
                     attention_mask=torch.ones(1, end, dtype=torch.long),
                     position_ids=p[None, None].expand(3, 1, -1), cache_position=p)
            chunks.append(out.last_hidden_state)
            past = out.past_key_values
    torch.testing.assert_close(torch.cat(chunks, 1), full, atol=1e-5, rtol=1e-4)


def test_face_mtp_cannot_see_residual_target():
    model = tiny_adapter("semantic_gan").eval()
    hidden, codes = torch.randn(2, 32), torch.randint(0, 16, (2, 5))
    changed = codes.clone()
    changed[:, 2:] = (changed[:, 2:] + 1) % 16
    with torch.no_grad():
        a, b = model.face_mtp.logits(hidden, codes), model.face_mtp.logits(hidden, changed)
    torch.testing.assert_close(a[0], b[0])


def test_padding_and_overlap_loss_masks():
    model = tiny_adapter("semantic_gan").eval()
    long, short = sample(model.geometry, loss=1), sample(model.geometry, blocks=1)
    short["history"] = []
    with torch.no_grad():
        a, b, together = model([long]), model([short]), model([long, short])
    assert a["supervised_blocks"] == 2
    assert b["supervised_blocks"] == 1
    torch.testing.assert_close(together["loss"], (2 * a["loss"] + b["loss"]) / 3, atol=1e-5, rtol=1e-5)


def test_audio_schedule_matches_official_generate():
    model = tiny_adapter("vae_vq").eval()
    hidden = torch.randn(2, 32)
    with torch.no_grad():
        primary = model.talker.codec_head(hidden).argmax(-1)
        inputs = torch.cat((hidden[:, None], model.talker.get_input_embeddings()(primary[:, None])), 1)
        rest = model.talker.code_predictor.generate(inputs_embeds=inputs, max_new_tokens=15,
                    min_new_tokens=15, do_sample=False, eos_token_id=None, pad_token_id=0)
        expected = torch.cat((primary[:, None], rest), -1)
        actual = model.predict_audio(hidden, lambda logits: logits.argmax(-1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_corrupt_cache_is_rebuilt_and_contract_changes_are_rejected(tmp_path):
    from qwen_tts.holdlistenstream.prepare import save_npz, verified
    from qwen_tts.holdlistenstream.files import write_json, read_contract, digest
    path = tmp_path / "sample.npz"
    save_npz(path, "build-one", {"audio": np.zeros((5, 16), dtype=np.uint16)}, {})
    assert verified(path, "build-one")
    assert not verified(path, "another-build")
    path.with_suffix(".npz.json").write_text("{interrupted")
    assert not verified(path, "build-one")
    save_npz(path, "build-one", {"audio": np.ones((5, 16), dtype=np.uint16)}, {})
    assert verified(path, "build-one")
    path.write_bytes(b"corrupted payload")
    assert not verified(path, "build-one")
    contract = {"version": 1, "clock": .4}
    contract["fingerprint"] = digest(contract)
    write_json(tmp_path / "contract.json", contract)
    assert read_contract(tmp_path / "contract.json")["clock"] == .4
    contract["clock"] = .5
    write_json(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="fingerprint"):
        read_contract(tmp_path / "contract.json")
