# SPDX-License-Identifier: Apache-2.0
"""Damage-based routed-expert widths: the allocator, and the measurement in a real build."""

import json
import logging

import omlx.oq as oq
from tests.test_oqt_streaming import checkpoint  # noqa: F401  (pytest fixture)

SHAPE = (8, 128, 256)   # (experts, out, in) of one fused expert projection


def _shapes(layers=4):
    shapes = {}
    for li in range(layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shapes[f"model.layers.{li}.mlp.switch_mlp.{proj}"] = SHAPE
        shapes[f"model.layers.{li}.self_attn.q_proj"] = (256, 256)
    return shapes


def _bytes(shapes, boost_map, base=5):
    total = 0
    for path, shape in shapes.items():
        if not oq._is_routed_expert(path):
            continue
        bits = (boost_map.get(path) or {}).get("bits", base)
        total += oq._tensor_quantized_bytes(shape, bits, 64, "affine")
    return total


def test_widths_follow_damage_at_the_same_bytes():
    shapes = _shapes()
    # Layer 0 is hurt most by fewer bits, layer 3 barely at all.
    damage = {
        0: {4: 0.080, 5: 0.020, 6: 0.005},
        1: {4: 0.010, 5: 0.004, 6: 0.002},
        2: {4: 0.010, 5: 0.004, 6: 0.002},
        3: {4: 0.0011, 5: 0.0010, 6: 0.0009},
    }
    new_map, report = oq._allocate_expert_widths(shapes, {}, damage, 5, (4, 5, 6))
    assert report["layers"][0] == 6 and report["layers"][3] == 4, report
    assert report["damage_after"] < report["damage_before"]
    assert _bytes(shapes, new_map) <= _bytes(shapes, {})
    # Non-expert tensors are never touched.
    assert not any("self_attn" in p for p in new_map)
    # Every projection of a layer gets the layer's width.
    assert {new_map[f"model.layers.0.mlp.switch_mlp.{p}"]["bits"] for p in ("gate_proj", "up_proj", "down_proj")} == {6}


def test_uniform_damage_keeps_the_plan():
    shapes = _shapes()
    damage = {li: {4: 0.01, 5: 0.004, 6: 0.002} for li in range(4)}
    new_map, report = oq._allocate_expert_widths(shapes, {}, damage, 5, (4, 5, 6))
    assert new_map == {} and report["layers"] == {}


def test_never_below_the_floor_and_noisy_damage_is_monotone():
    assert oq._expert_alloc_widths(5) == (4, 5, 6)
    assert min(oq._expert_alloc_widths(4)) >= oq._EXPERT_ALLOC_MIN_BITS
    shapes = _shapes(2)
    # Noise says 6 bits is worse than 5 for layer 1: treated as no better.
    damage = {0: {4: 0.05, 5: 0.01, 6: 0.002}, 1: {4: 0.001, 5: 0.0009, 6: 0.003}}
    new_map, report = oq._allocate_expert_widths(shapes, {}, damage, 5, (4, 5, 6))
    assert report["layers"] == {0: 6, 1: 4}, report


def test_layers_without_a_measurement_or_with_a_plan_boost_are_kept():
    shapes = _shapes(3)
    plan = {"model.layers.2.mlp.switch_mlp.down_proj": {"bits": 6, "group_size": 64, "mode": "affine"}}
    damage = {0: {4: 0.05, 5: 0.01, 6: 0.002}, 2: {4: 0.001, 5: 0.0009, 6: 0.0008}}
    new_map, report = oq._allocate_expert_widths(shapes, plan, damage, 5, (4, 5, 6))
    # Only layer 0 is eligible: nothing to trade against, so the plan stands.
    assert new_map == plan and report["measured_layers"] == 1


def test_a_streamed_oqt_build_measures_damage_and_plans_widths(checkpoint, tmp_path, caplog):  # noqa: F811
    """oQ5 on the tiny Qwen4-Exp checkpoint, sensitivity measured in the streamed sweep:
    every MoE layer gets a damage measurement, the plan is decided from it, and the
    written experts use only widths from 4..6."""
    source, _ = checkpoint
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger="omlx.oq"):
        oq.quantize_oq_streaming(
            str(source), str(out), 5, group_size=32, enhanced=True, stream_calibration=True,
            imatrix_num_samples=4, imatrix_seq_length=8, teacher_matched=True,
            oqt_cache_path=str(tmp_path / "cache"))
    lines = [r.getMessage() for r in caplog.records if "expert widths" in r.getMessage()]
    assert lines and "no per-layer expert damage" not in lines[0], lines
    # Unquantizable expert shapes (the tiny down_proj) are never traded for free bits.
    assert "expert bytes" not in lines[0] or "->" in lines[0]
    cfg = json.loads((out / "config.json").read_text())
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    expert_bits = {v["bits"] for k, v in q.items() if isinstance(v, dict) and oq._is_routed_expert(k)}
    assert expert_bits <= {4, 5, 6}, q


def test_allocation_can_be_switched_off(checkpoint, tmp_path, caplog):  # noqa: F811
    source, _ = checkpoint
    with caplog.at_level(logging.INFO, logger="omlx.oq"):
        oq.quantize_oq_streaming(
            str(source), str(tmp_path / "out"), 5, group_size=32, enhanced=True, stream_calibration=True,
            imatrix_num_samples=4, imatrix_seq_length=8, teacher_matched=True,
            oqt_cache_path=str(tmp_path / "cache"), expert_allocation=False)
    assert not any("expert widths" in r.getMessage() for r in caplog.records)
