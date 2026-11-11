# SPDX-License-Identifier: Apache-2.0
"""Tests for oQ-T milestone 1: the GPTQ core in MLX affine layout."""

import mlx.core as mx
import numpy as np
import pytest

from omlx import oqt
from omlx.oq import _weighted_affine_quantize


def _correlated_inputs(n_tokens, in_dim, seed=0, outliers=4):
    """Activations with correlated channels and a few outlier channels,
    the regime where full-Hessian error feedback should beat diagonal-only."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_tokens, in_dim // 4)).astype(np.float32)
    mix = rng.standard_normal((in_dim // 4, in_dim)).astype(np.float32)
    x = z @ mix + 0.1 * rng.standard_normal((n_tokens, in_dim)).astype(np.float32)
    x[:, :outliers] *= 20.0
    return mx.array(x)


def _hessian(x):
    return (x.T @ x) / x.shape[0]


def _deq(packed, scales, biases, group_size, bits):
    return mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits).astype(mx.float32)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_layout_matches_mx_quantize(bits, group_size):
    w = mx.random.normal((96, 256), key=mx.random.key(1)).astype(mx.bfloat16)
    h = _hessian(_correlated_inputs(512, 256))
    packed, scales, biases = oqt.gptq_quantize(w, h, bits, group_size)
    ref_p, ref_s, ref_b = mx.quantize(w, group_size=group_size, bits=bits)
    assert packed.shape == ref_p.shape and packed.dtype == ref_p.dtype
    assert scales.shape == ref_s.shape and scales.dtype == mx.bfloat16
    assert biases.shape == ref_b.shape and biases.dtype == mx.bfloat16
    deq = _deq(packed, scales, biases, group_size, bits)
    assert not bool(mx.any(mx.isnan(deq)).item())
    # Sanity: reconstruction is in the right range for the bit width.
    rel = float(mx.mean((deq - w.astype(mx.float32)) ** 2).item()) / float(
        mx.mean(w.astype(mx.float32) ** 2).item())
    assert rel < {2: 0.5, 3: 0.2, 4: 0.06, 5: 0.02, 6: 0.006, 8: 0.001}[bits]


@pytest.mark.parametrize("bits", [3, 4])
def test_gptq_beats_oqe_on_output_error(bits):
    gs = 64
    w = (mx.random.normal((256, 512), key=mx.random.key(2)) * 0.02).astype(mx.bfloat16)
    x = _correlated_inputs(2048, 512, seed=3)
    h = _hessian(x)
    imp = mx.diagonal(h)

    qp, qs, qb = _weighted_affine_quantize(w, gs, bits, imp)
    err_oqe = oqt.output_error(w, _deq(qp, qs, qb, gs, bits), h)

    gp, gsc, gb = oqt.gptq_quantize(w, h, bits, gs)
    err_gptq = oqt.output_error(w, _deq(gp, gsc, gb, gs, bits), h)

    assert err_gptq < 0.8 * err_oqe, (bits, err_gptq, err_oqe)


def test_uncorrelated_inputs_do_not_regress():
    """With an identity Hessian there is nothing to compensate; GPTQ must not
    be meaningfully worse than oQe's clipping search."""
    gs, bits = 64, 4
    w = (mx.random.normal((128, 256), key=mx.random.key(4)) * 0.02).astype(mx.bfloat16)
    h = mx.eye(256)
    qp, qs, qb = _weighted_affine_quantize(w, gs, bits, mx.ones((256,)))
    gp, gsc, gb = oqt.gptq_quantize(w, h, bits, gs)
    e_oqe = oqt.output_error(w, _deq(qp, qs, qb, gs, bits), h)
    e_gptq = oqt.output_error(w, _deq(gp, gsc, gb, gs, bits), h)
    assert e_gptq <= 1.05 * e_oqe


def test_static_reproduces_oqe_with_identity_hessian():
    gs, bits = 64, 4
    w = (mx.random.normal((64, 256), key=mx.random.key(13)) * 0.02).astype(mx.bfloat16)
    qp, qs, qb = _weighted_affine_quantize(w, gs, bits, mx.ones((256,)))
    gp, gsc, gb = oqt.gptq_quantize(w, mx.eye(256), bits, gs, group_params="static")
    np.testing.assert_array_equal(np.array(gp), np.array(qp))
    np.testing.assert_array_equal(np.array(gsc.astype(mx.float32)), np.array(qs.astype(mx.float32)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_is_never_worse_than_either(seed):
    gs, bits = 64, 3
    w = (mx.random.normal((4, 64, 256), key=mx.random.key(20 + seed)) * 0.02).astype(mx.bfloat16)
    hs = mx.stack([_hessian(_correlated_inputs(512, 256, seed=seed * 10 + e, outliers=e))
                   for e in range(4)])
    errs = {}
    for mode in ("static", "dynamic", "best"):
        p, s, b = oqt.gptq_quantize(w, hs, bits, gs, group_params=mode)
        deq = _deq(p, s, b, gs, bits)
        errs[mode] = [oqt.output_error(w[e], deq[e], hs[e]) for e in range(4)]
    for e in range(4):
        assert errs["best"][e] <= min(errs["static"][e], errs["dynamic"][e]) * (1 + 1e-4)


def test_batched_equals_per_expert():
    gs, bits = 32, 4
    w = (mx.random.normal((4, 64, 128), key=mx.random.key(5)) * 0.02).astype(mx.bfloat16)
    hs = mx.stack([_hessian(_correlated_inputs(256, 128, seed=s)) for s in range(4)])
    bp, bs, bb = oqt.gptq_quantize(w, hs, bits, gs)
    for e in range(4):
        p, s, b = oqt.gptq_quantize(w[e], hs[e], bits, gs)
        np.testing.assert_allclose(
            np.array(_deq(bp[e], bs[e], bb[e], gs, bits)),
            np.array(_deq(p, s, b, gs, bits)), atol=1e-6)


def test_dead_inputs_and_singular_hessian():
    gs, bits = 32, 4
    w = (mx.random.normal((32, 128), key=mx.random.key(6)) * 0.02).astype(mx.bfloat16)
    x = _correlated_inputs(64, 128, seed=7)
    x[:, 10:20] = 0.0                       # dead channels
    h = _hessian(x)                         # rank-deficient: 64 tokens < 128 dims
    p, s, b = oqt.gptq_quantize(w, h, bits, gs)
    deq = _deq(p, s, b, gs, bits)
    assert not bool(mx.any(mx.isnan(deq)).item())
    # Dead channels see zero input, so their values cannot affect the output;
    # what matters is that the live channels are still quantized well.
    assert oqt.output_error(w, deq, h) < 0.05


def test_routed_accumulator_matches_bruteforce():
    E, T, D, K = 6, 40, 16, 2
    x = mx.random.normal((T, D), key=mx.random.key(8))
    ids = mx.random.randint(0, E, (T, K), key=mx.random.key(9))
    acc = oqt.HessianAccumulator(D, batch=E)
    acc.add_routed(x, ids)
    xn, idn = np.array(x), np.array(ids)
    for e in range(E):
        rows = [t for t in range(T) for k in range(K) if idn[t, k] == e]
        ref = sum((np.outer(xn[t], xn[t]) for t in rows), np.zeros((D, D)))
        np.testing.assert_allclose(np.array(acc.sum_xxT[e]), ref, rtol=1e-4, atol=1e-4)
        assert float(acc.count[e].item()) == len(rows)


def test_shrinkage_unseen_expert_gets_layer_average():
    D = 8
    s = mx.zeros((3, D, D))
    s[0] = mx.eye(D) * 1000.0               # 1000 tokens of identity energy
    s[1] = mx.eye(D) * 2.0 * 10.0           # 10 tokens, 2x energy
    counts = mx.array([1000.0, 10.0, 0.0])
    h = oqt.shrink_expert_hessians(s, counts, prior_tokens=100.0)
    h_bar = (s.sum(axis=0) / counts.sum())
    np.testing.assert_allclose(np.array(h[2]), np.array(h_bar), rtol=1e-5)
    # The well-covered expert stays close to its own statistics.
    np.testing.assert_allclose(np.array(h[0]), np.eye(D), rtol=0.02)


def test_experts_wrapper_matches_manual_chunks():
    gs, bits = 32, 4
    E, D = 5, 64
    w = (mx.random.normal((E, 32, D), key=mx.random.key(10)) * 0.02).astype(mx.bfloat16)
    x = mx.random.normal((200, D), key=mx.random.key(11))
    ids = mx.random.randint(0, E, (200, 2), key=mx.random.key(12))
    acc = oqt.HessianAccumulator(D, batch=E)
    acc.add_routed(x, ids)
    p, s, b = oqt.gptq_quantize_experts(w, acc.sum_xxT, acc.count, bits, gs, expert_chunk=2)
    assert p.shape[0] == E
    h = oqt.shrink_expert_hessians(acc.sum_xxT, acc.count)
    p2, s2, b2 = oqt.gptq_quantize(w, h, bits, gs)
    np.testing.assert_allclose(np.array(_deq(p, s, b, gs, bits)),
                               np.array(_deq(p2, s2, b2, gs, bits)), atol=1e-6)
