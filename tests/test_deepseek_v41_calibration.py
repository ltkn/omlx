"""V4.1 mixed-format projections expose their real calibration inputs."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.oq import OQImatrixCollector
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import Expert
from omlx.patches.deepseek_v41.quantization import QuantizedProjection


@pytest.mark.parametrize("length", [2, 32])
def test_capture_dense_and_routed_quantized_projection_inputs(length):
    mx.random.seed(908)
    model = nn.Module()
    model.dense = QuantizedProjection(
        *mx.quantize(
            mx.random.normal((64, 64)).astype(mx.bfloat16) * 0.1,
            bits=8,
            group_size=32,
            mode="mxfp8",
        ),
        8,
        "mxfp8",
    )
    model.expert = Expert(
        ModelConfig(dim=64, moe_inter_dim=64, n_routed_experts=4), True
    )
    for name in ("w1", "w3", "w2"):
        w, s = mx.quantize(
            mx.random.normal((4, 64, 64)).astype(mx.bfloat16) * 0.1,
            bits=4,
            group_size=32,
            mode="mxfp4",
        )
        setattr(model.expert, name, QuantizedProjection(w, s, 4, "mxfp4"))
    x = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
    ids = mx.broadcast_to(mx.array([[[0, 2]]], mx.uint32), (1, length, 2))
    weights = mx.broadcast_to(mx.array([[[0.3, 0.7]]]), ids.shape)

    def forward():
        value = model.dense(x)
        return model.expert(value[..., None, None, :], ids, weights)

    expected = forward()
    mx.eval(expected)
    original_dense = model.dense
    collector = OQImatrixCollector()
    try:
        assert collector.install(model) == 4
        actual = forward()
        mx.eval(actual)
        assert set(collector.entries) == {
            "dense",
            "expert.w1",
            "expert.w3",
            "expert.w2",
        }
        np.testing.assert_array_equal(collector.entries["dense"].counts, [length])
        np.testing.assert_allclose(
            collector.entries["dense"].in_sum2,
            np.square(np.asarray(x.astype(mx.float32))).sum(axis=(0, 1)),
            rtol=1e-6,
        )
        for name in ("w1", "w3", "w2"):
            entry = collector.entries["expert." + name]
            assert entry.in_sum2.shape == (4, 64)
            np.testing.assert_array_equal(entry.counts, [length, 0, length, 0])
            assert np.isfinite(entry.in_sum2).all()
        np.testing.assert_allclose(
            actual.astype(mx.float32),
            expected.astype(mx.float32),
            rtol=0.03,
            atol=0.0002,
        )
    finally:
        collector.restore(model)
    assert model.dense is original_dense


def test_inventory_matches_streamed_projection_shapes_and_bytes(tmp_path):
    import json

    from test_deepseek_v41 import write_checkpoint

    from omlx.patches.deepseek_v41.convert import iter_source_weights
    from omlx.patches.deepseek_v41.oq_inventory import projection_inventory

    source, _ = write_checkpoint(tmp_path)
    config = json.loads((source / "config.json").read_text())
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    inventory = projection_inventory(source, config, mapping)
    visited = set()
    for tensors, specs in iter_source_weights(source, config, mapping):
        for key, weight in tensors.items():
            if not key.endswith(".weight"):
                continue
            name = key.removesuffix(".weight")
            shape = list(weight.shape)
            if name in specs:
                shape[-1] = shape[-1] * 32 // specs[name]["bits"]
            assert tuple(shape) == inventory[name]["shape"]
            assert (
                sum(t.nbytes for k, t in tensors.items() if k.startswith(name + "."))
                == inventory[name]["source_bytes"]
            )
            visited.add(name)
    assert visited == set(inventory)


def test_calibration_resume_matches_single_pass(monkeypatch):
    import omlx.oq as oq
    from omlx.patches.deepseek_v41.calibration import collect_imatrix

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 8)

        def make_cache(self):
            return []

        def __call__(self, x, cache):
            return self.proj(x.astype(mx.float32))

    model = nn.Module()
    model.language_model = Core()
    samples = mx.arange(16).reshape(4, 4)
    monkeypatch.setattr(
        oq, "_load_calibration_data", lambda tokenizer, dataset, n, length: samples[:n]
    )
    full, _ = collect_imatrix(model, None, num_samples=4, seq_length=4)
    first, metadata = collect_imatrix(model, None, num_samples=2, seq_length=4)
    original = first["language_model.proj"].counts.copy()
    progress = []
    resumed, _ = collect_imatrix(
        model,
        None,
        num_samples=4,
        seq_length=4,
        initial=oq.OQImatrixData(first, metadata, path="fixture"),
        progress=lambda done, total, entries: progress.append(done),
    )
    assert progress == [3, 4]
    np.testing.assert_array_equal(first["language_model.proj"].counts, original)
    for name in full:
        np.testing.assert_array_equal(full[name].counts, resumed[name].counts)
        np.testing.assert_array_equal(full[name].in_sum2, resumed[name].in_sum2)


def test_official_collector_routes_v41_settings(monkeypatch):
    from omlx import oq
    from omlx.patches.deepseek_v41 import calibration

    received = {}

    def collect(path, **kwargs):
        received.update(path=path, **kwargs)
        return {}, {"marker": "v41"}

    monkeypatch.setattr(calibration, "collect_checkpoint_imatrix", collect)

    def callback(*args):
        pass

    assert oq._collect_imatrix(
        "source",
        {"model_type": "deepseek_v41"},
        calib_dataset="custom",
        num_samples=37,
        seq_length=256,
        progress_callback=callback,
        progress_start=7,
        progress_end=19,
    ) == ({}, {"marker": "v41"})
    assert received == dict(
        path="source",
        calib_dataset="custom",
        num_samples=37,
        seq_length=256,
        progress_callback=callback,
        progress_start=7,
        progress_end=19,
    )


def test_official_v41_collector_closes_model_on_cancel(monkeypatch):
    from types import SimpleNamespace

    from omlx.patches.deepseek_v41 import calibration, loading

    events = []
    model = SimpleNamespace(close=lambda: events.append("close"))

    def load(path, **kwargs):
        assert kwargs == {"engram_ssd_offload": True}
        return model, SimpleNamespace(tokenizer="tokenizer")

    def collect(model, tokenizer, **kwargs):
        assert tokenizer == "tokenizer"
        kwargs["progress"](1, 2, {})

    def cancel(*args):
        raise RuntimeError("cancelled")

    monkeypatch.setattr(loading, "load", load)
    monkeypatch.setattr(calibration, "collect_imatrix", collect)
    monkeypatch.setattr(calibration.mx, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(calibration.mx, "clear_cache", lambda: events.append("clear"))
    with pytest.raises(RuntimeError, match="cancelled"):
        calibration.collect_checkpoint_imatrix(
            "source",
            calib_dataset="custom",
            num_samples=2,
            seq_length=512,
            progress_callback=cancel,
        )
    assert events == ["close", "sync", "clear"]


@pytest.mark.parametrize("sufficient_at, expected", [(2, 2), (4, 4), (99, 6)])
def test_adaptive_calibration_stops_on_coverage_or_available_data(
    monkeypatch, sufficient_at, expected
):
    import omlx.oq as oq
    from omlx.patches.deepseek_v41.calibration import collect_imatrix

    class Collector:
        switch_capture_modules = 1

        def __init__(self):
            self.entries = {"expert": object()}

        def install(self, model):
            return 1

        def restore(self, model):
            model.restored = True

    class Core:
        calls = 0

        def make_cache(self):
            return []

        def __call__(self, x, cache):
            self.calls += 1
            return x

    from types import SimpleNamespace

    model = SimpleNamespace(language_model=Core(), restored=False)
    requested = []

    def samples(tokenizer, dataset, number, length):
        requested.append(number)
        return np.arange(12).reshape(6, 2)

    monkeypatch.setattr(oq, "OQImatrixCollector", Collector)
    monkeypatch.setattr(oq, "_load_calibration_data", samples)
    monkeypatch.setattr(
        oq, "_imatrix_expert_coverage_stats", lambda entries: {"fixture": True}
    )
    monkeypatch.setattr(
        oq,
        "_imatrix_expert_coverage_sufficient",
        lambda stats, **kwargs: model.language_model.calls >= sufficient_at,
    )
    # This test covers the sample scheduling and restoration, not GPU arithmetic.
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    steps = []
    _, metadata = collect_imatrix(
        model,
        None,
        num_samples=2,
        seq_length=2,
        adaptive=True,
        progress=lambda done, total, entries: steps.append(done),
    )
    assert requested == [16]
    assert steps == list(range(1, expected + 1))
    assert metadata["processed_samples"] == expected
    assert metadata["adaptive_max_samples"] == 6
    assert model.restored


@pytest.mark.parametrize("vision", ["vision.proj", "aligner.proj", "image_projection"])
@pytest.mark.parametrize("calibrated_vision", [False, True])
@pytest.mark.parametrize(
    "level, target, cap, bits", [(3, 3.5, 3.7, 3), (2, 2.8, 3.0, 2)]
)
def test_affine_plan_preserves_vision_precision(
    vision, calibrated_vision, level, target, cap, bits
):
    from types import SimpleNamespace

    from omlx.patches.deepseek_v41.oq_inventory import build_affine_plan

    good = "language_model.layers.0.ffn.experts.w1"
    missing_expert = "language_model.layers.1.ffn.experts.w1"
    inventory = {
        good: dict(shape=(128, 64, 64), source_bytes=278528),
        missing_expert: dict(shape=(2, 64, 64), source_bytes=4352),
        vision: dict(shape=(64, 64), source_bytes=8192),
    }
    imatrix = SimpleNamespace(
        entries={
            good: SimpleNamespace(counts=np.ones(128)),
            missing_expert: SimpleNamespace(counts=np.array([10, 0])),
        }
    )
    if calibrated_vision:
        imatrix.entries[vision] = SimpleNamespace(counts=np.ones(1))
    budget = dict(
        remaining_weights=dict(logical_parameters=536576, tensor_bytes=291072)
    )
    specs, report = build_affine_plan(
        inventory,
        budget,
        {"model_type": "deepseek_v41"},
        imatrix,
        {0: 0.25, 1: 0.5},
        oq_level=level,
        target=target,
        cap=cap,
    )
    assert set(specs) == {good, missing_expert}
    assert specs[good]["bits"] == bits
    assert specs[missing_expert]["bits"] == bits
    assert report["fixed_source_bytes"] == 8192
    # Affine group-64 storage costs bits/8 bytes plus a BF16 scale and bias.
    assert report["tensor_bytes"] == 8192 + (524288 + 8192) * (2 * bits + 1) // 16
    assert report["effective_bpw"] <= cap
    assert report["unobserved_experts"] == {missing_expert: [1]}
    assert report["unobserved_expert_policy"] == "uniform_importance_at_allocated_bits"
    assert report["preserved_modules"][vision] == "precision_policy"
    with pytest.raises(ValueError, match="above.*cap"):
        build_affine_plan(
            inventory,
            budget,
            {"model_type": "deepseek_v41"},
            imatrix,
            {0: 0.25, 1: 0.5},
            oq_level=3,
            target=3.0,
            cap=3.1,
        )


def test_affine_plan_forwards_legacy_allocation(monkeypatch):
    from types import SimpleNamespace

    from omlx import oq
    from omlx.patches.deepseek_v41.oq_inventory import build_affine_plan

    expert = "language_model.layers.0.ffn.experts.w1"
    inventory = {expert: dict(shape=(128, 64, 64), source_bytes=278528)}
    imatrix = SimpleNamespace(entries={expert: SimpleNamespace(counts=np.ones(128))})
    budget = dict(
        remaining_weights=dict(logical_parameters=524288, tensor_bytes=278528)
    )
    seen = {}
    real = oq._build_quant_plan

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(oq, "_build_quant_plan", spy)
    build_affine_plan(
        inventory,
        budget,
        {"model_type": "deepseek_v41"},
        imatrix,
        {"0": 0.5},
        oq_level=3,
        target=3.5,
        cap=3.7,
        legacy_allocation=True,
    )
    # V4.1 keeps its own exporter constraints but opts into the fixed Q8
    # mandatory patterns through the shared allocator.
    assert seen["legacy_allocation"] is True
    assert seen["always_on_rules"] is False


@pytest.mark.parametrize(
    "model_type,collection,missing",
    [
        ("deepseek_v41", {"uncalibrated_policy": "preserve_source_precision"}, False),
        ("deepseek_v41", None, True),
        ("deepseek_v41", {}, True),
        ("qwen4_exp", {"uncalibrated_policy": "preserve_source_precision"}, True),
    ],
)
def test_mtp_cache_reuse_requires_v41_preservation_policy(
    monkeypatch, model_type, collection, missing
):
    from omlx import oq
    from omlx.utils import model_loading

    monkeypatch.setattr(model_loading, "_has_mtp_heads", lambda config: True)
    monkeypatch.setattr(model_loading, "_checkpoint_has_mtp_weights", lambda path: True)
    cache = oq.OQImatrixData({}, {"collection": collection}, "fixture")
    assert (
        oq._oqe_cache_missing_mtp_entries(cache, {"model_type": model_type}, "source")
        is missing
    )


def test_streaming_calibration_matches_resident_with_engram(tmp_path, monkeypatch):
    import json

    import mlx.core as mx
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    from test_deepseek_v41 import write_checkpoint

    from omlx import oq
    from omlx.patches.deepseek_v41.calibration import (
        collect_checkpoint_imatrix_streaming,
        collect_imatrix,
    )
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41.sensitivity import measure_sensitivity

    source, _ = write_checkpoint(
        tmp_path,
        vision=False,
        dim=64,
        moe_inter_dim=64,
        engram_layer_ids=(1, 3),
        engram_num_embeddings=(72, 204),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    config = json.loads((source / "config.json").read_text())
    tokens = mx.array(
        [
            [3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14],
        ]
    )
    monkeypatch.setattr(
        oq,
        "_load_calibration_data",
        lambda tokenizer, dataset, num_samples, seq_length: tokens[:num_samples],
    )
    with mx.stream(mx.cpu):
        model, processor = load(source, engram_ssd_offload=True)
        try:
            resident, _ = collect_imatrix(
                model,
                processor.tokenizer,
                num_samples=2,
                seq_length=6,
                calib_dataset="fixture",
            )
            resident_sensitivity = measure_sensitivity(
                model,
                processor.tokenizer,
                bits=2,
                num_samples=2,
                seq_length=6,
                calib_dataset="fixture",
            )
        finally:
            model.close()
            del model, processor
            mx.clear_cache()

        streamed, metadata = collect_checkpoint_imatrix_streaming(
            source,
            load_tokenizer(source),
            config,
            calib_dataset="fixture",
            num_samples=2,
            seq_length=6,
            calib_data=tokens,
            measure_sensitivity=True,
            sensitivity_oq_level=2.5,
        )

    assert set(streamed) == set(resident)
    for name in resident:
        np.testing.assert_array_equal(streamed[name].counts, resident[name].counts)
        np.testing.assert_allclose(
            streamed[name].in_sum2, resident[name].in_sum2, rtol=1e-5, atol=1e-5
        )
    assert metadata["load_kind"] == "streaming"
    assert metadata["sensitivity_map"].keys() == resident_sensitivity.keys()
    for layer, score in resident_sensitivity.items():
        assert metadata["sensitivity_map"][layer] == pytest.approx(score, abs=1e-7)
