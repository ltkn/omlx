# SPDX-License-Identifier: Apache-2.0
"""oQ-T milestone 2: the streamed sweep, end to end on a tiny Qwen4-Exp
checkpoint (the same fixture shape as test_oq.py::TestStreamedCalibration)."""

import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    from mlx.utils import tree_flatten
    from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config
    import omlx.oq as oq
    from mlx_vlm.models.qwen4_exp.language import configure_mtp_runtime, configure_ple_runtime
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    cfg = _tiny_config()
    cfg.text_config.linear_key_head_dim = 32
    cfg.text_config.linear_value_head_dim = 32
    source = tmp_path / "source"
    source.mkdir()
    index = source / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}))
    configure_ple_runtime(source, mode="resident")
    configure_mtp_runtime(source, enabled=True)
    mx.random.seed(7)
    model = Model(cfg)
    model.eval()
    weights = {k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v
               for k, v in tree_flatten(model.parameters())}
    scale_key = next(k for k in weights if k.endswith("ngram_embedding.weight_scale"))
    weights[scale_key] = mx.array([0.25], dtype=mx.bfloat16)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    config = asdict(cfg)
    config["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    config["text_config"]["mtp_num_hidden_layers"] = 1
    (source / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    index.write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in weights}}))
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9],
                       [3, 4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10, 11]], dtype=mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    yield source, tokens
    configure_mtp_runtime(source, enabled=False)
    configure_ple_runtime(source, mode="resident")
    mx.synchronize()
    mx.clear_cache()


def _build(source, out, cache, **extra):
    import omlx.oq as oq

    oq.quantize_oq_streaming(
        str(source), str(out), 4, group_size=32, enhanced=True, stream_calibration=True,
        imatrix_num_samples=4, imatrix_seq_length=8, sensitivity_map_override={0: 1, 1: 1},
        teacher_matched=True, oqt_cache_path=str(cache), **extra)


def _load_all(path):
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        out.update(mx.load(str(f)))
    return out


def test_oqt_build_loads_uses_cache_and_resumes(checkpoint, tmp_path):
    from mlx_vlm.utils import load_model
    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime

    source, tokens = checkpoint
    cache = tmp_path / "oqt-cache"
    out1 = tmp_path / "out1"
    _build(source, out1, cache)

    # 1. The output is a normal oQ checkpoint that loads and runs.
    configure_qwen4_exp_runtime(out1, mode="resident", mtp_enabled=False)
    loaded = load_model(out1, lazy=True, strict=True)
    assert mx.isfinite(loaded(tokens).logits).all().item()

    # 2. The writer used the sweep's tensors, byte for byte.
    report = json.loads((out1 / "oqt_report.json").read_text())
    used = report["used"]
    assert any(".mlp.switch_mlp." in u for u in used), used
    assert any(".self_attn." in u or ".linear_attn." in u for u in used), used
    assert report["declined_by_writer"] == []
    written = _load_all(out1)
    cached = {}
    for f in sorted(cache.glob("layer-*.safetensors")):
        cached.update(mx.load(str(f)))
    for base in used:
        np.testing.assert_array_equal(np.array(written[f"{base}.weight"]),
                                      np.array(cached[f"{base}.weight"]))

    # 3. Every quantized tensor recorded both errors; oQ-T is not worse overall.
    s = report["summary"]
    assert s["tensors_compared"] == len(used)
    assert s["median_err_ratio_oqt_vs_oqe"] < 1.05

    # 4. A rerun with the same cache reuses every layer and writes the same bytes.
    out2 = tmp_path / "out2"
    _build(source, out2, cache)
    report2 = json.loads((out2 / "oqt_report.json").read_text())
    assert all(layer.get("reused") for layer in report2["layers"].values())
    written2 = _load_all(out2)
    for base in used:
        np.testing.assert_array_equal(np.array(written[f"{base}.weight"]),
                                      np.array(written2[f"{base}.weight"]))


def test_oqt_cache_invalidates_on_setting_change(checkpoint, tmp_path):
    source, _ = checkpoint
    cache = tmp_path / "oqt-cache"
    _build(source, tmp_path / "a", cache)
    _build(source, tmp_path / "b", cache, oqt_options={"damp": 0.05})
    report = json.loads((tmp_path / "b" / "oqt_report.json").read_text())
    assert not any(layer.get("reused") for layer in report["layers"].values())


def test_deepseek_v41_is_refused_clearly():
    from omlx import oqt

    with pytest.raises(NotImplementedError, match="DeepSeek V4.1"):
        oqt.run_streaming_sweep("/nonexistent", {"model_type": "deepseek_v41"}, 4, 64,
                                cache_dir="/tmp/unused-oqt", calib_dataset="x")


def test_exact_script_flags_with_mtp_and_ngram8(checkpoint, tmp_path):
    """The flags scripts/omlx_oqt.sh passes: MTP kept, n-gram table at 8 bits,
    imatrix cache reused, calibration from a file path."""
    from mlx_vlm.utils import load_model
    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
    import omlx.oq as oq

    source, tokens = checkpoint
    bundle = tmp_path / "bundle.txt"
    bundle.write_text("calibration text\n")
    out = tmp_path / "out"
    kwargs = dict(
        enhanced=True, stream_calibration=True, teacher_matched=True,
        oqt_cache_path=str(tmp_path / "cache"), preserve_mtp=True, ngram_bits=8,
        calib_dataset=str(bundle), sensitivity_calib_dataset=str(bundle),
        imatrix_cache_path=str(tmp_path / "imatrix.npz"), imatrix_reuse_cache=True,
        imatrix_num_samples=4, imatrix_seq_length=8, group_size=32,
        sensitivity_map_override={0: 1, 1: 1},
    )
    oq.quantize_oq_streaming(str(source), str(out), 4, **kwargs)
    configure_qwen4_exp_runtime(out, mode="resident", mtp_enabled=True)
    loaded = load_model(out, lazy=True, strict=True)
    assert mx.isfinite(loaded(tokens).logits).all().item()
    report = json.loads((out / "oqt_report.json").read_text())
    assert report["used"] and report["declined_by_writer"] == []
    # Second run (the script's resume path): imatrix and every oQ-T layer reused.
    oq.quantize_oq_streaming(str(source), str(tmp_path / "out2"), 4, **kwargs)
    report2 = json.loads((tmp_path / "out2" / "oqt_report.json").read_text())
    assert all(layer.get("reused") for layer in report2["layers"].values())
