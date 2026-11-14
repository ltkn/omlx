# SPDX-License-Identifier: Apache-2.0
"""oQ-T: teacher-matched quantization for oQ.

Milestone 1 — the core quantizer. Everything here produces MLX's standard
affine layout (``mx.quantize(..., mode="affine")``: packed uint32 codes plus
per-group scales and biases with ``w ≈ scale * q + bias``), so outputs load
and run with the stock kernels; only *which* codes and scales are chosen
changes.

What differs from oQe's ``_weighted_affine_quantize``:

* **Error feedback (GPTQ).** Columns are quantized left to right; the
  rounding error of each column is pushed onto the not-yet-quantized
  columns through the inverse Hessian of the layer input, so the *output*
  error ``||X Wᵀ - X Qᵀ||`` is minimized instead of each weight's own error.
  oQe sees only the Hessian diagonal (the imatrix); GPTQ uses the full
  ``XᵀX``, which captures correlated input channels.
* **Exact storage precision.** Scales and biases are rounded to the output
  dtype (bf16/fp16) *before* the codes are chosen, so the compensation is
  computed against the values that are actually stored.
* **Batched experts.** All experts of a MoE layer are quantized in one
  vectorized pass (``W: [E, out, in]``, ``H: [E, in, in]``).
* **Hessian shrinkage for rare experts.** An expert that saw few tokens has
  a noisy Hessian; :func:`shrink_expert_hessians` blends it toward the
  layer-wide average in proportion to how few tokens it saw, so coverage
  gaps degrade gracefully instead of producing garbage compensation.

Group parameters are chosen per group, when the sweep reaches the group,
from the *already error-updated* weights, with the same clipping search oQe
uses (weighted by the Hessian diagonal).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_VALID_BITS = (2, 3, 4, 5, 6, 8)
_CLIP_FACTORS = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25)


# ---------------------------------------------------------------------------
# Hessians
# ---------------------------------------------------------------------------


class HessianAccumulator:
    """Running ``Σ x xᵀ`` and token count for one input space.

    Dense (``batch == 0``): one ``[in, in]`` sum.

    Experts (``batch == E``): one ``[in, in]`` sum **per expert, stored as
    separate arrays** (never one stacked ``[E, in, in]`` tensor, which MLX
    would copy on every per-expert update), plus a running total for the
    layer-wide average used by shrinkage. Routed tokens are sorted by expert
    and each expert's slice is multiplied once, so the work is ``k·T·in²``
    rather than ``E·T·in²``.
    """

    def __init__(self, in_dim: int, batch: int = 0):
        self.in_dim = int(in_dim)
        self.batch = int(batch)
        if self.batch:
            self.expert_sums: list[mx.array | None] = [None] * self.batch
            self.total = mx.zeros((self.in_dim, self.in_dim), dtype=mx.float32)
            self.counts = np.zeros((self.batch,), dtype=np.float64)
        else:
            self.total = mx.zeros((self.in_dim, self.in_dim), dtype=mx.float32)
            self.counts = np.zeros((), dtype=np.float64)

    # -- dense ------------------------------------------------------------
    def add(self, x: mx.array) -> None:
        """Dense input ``x[..., in_dim]``."""
        x = x.reshape(-1, self.in_dim).astype(mx.float32)
        self.total = self.total + x.T @ x
        self.counts = self.counts + x.shape[0]
        mx.eval(self.total)

    # -- experts ----------------------------------------------------------
    def add_routed(self, x: mx.array, expert_ids: mx.array) -> None:
        """Token-level input ``x[T, in]`` routed to ``expert_ids[T, k]``."""
        x = x.reshape(-1, self.in_dim)
        ids = np.asarray(expert_ids).reshape(x.shape[0], -1).astype(np.int64)
        k = ids.shape[1]
        self._add_pairs(x, np.repeat(np.arange(x.shape[0]), k), ids.reshape(-1))

    def add_pairs(self, x_pairs: mx.array, expert_ids: mx.array) -> None:
        """Pair-level input (one row per (token, expert) pair), e.g. down_proj."""
        x_pairs = x_pairs.reshape(-1, self.in_dim)
        ids = np.asarray(expert_ids).reshape(-1).astype(np.int64)
        self._add_pairs(x_pairs, np.arange(x_pairs.shape[0]), ids)

    def _add_pairs(self, x, rows: np.ndarray, ids: np.ndarray) -> None:
        valid = (ids >= 0) & (ids < self.batch)
        rows, ids = rows[valid], ids[valid]
        if ids.size == 0:
            return
        order = np.argsort(ids, kind="stable")
        ids, rows = ids[order], rows[order]
        counts = np.bincount(ids, minlength=self.batch)
        starts = np.concatenate([[0], np.cumsum(counts)])
        xs = x[mx.array(rows)].astype(mx.float32)               # [P, in], sorted by expert
        self.total = self.total + xs.T @ xs
        pending = []
        for e in np.nonzero(counts)[0]:
            seg = xs[int(starts[e]) : int(starts[e + 1])]
            upd = seg.T @ seg
            cur = self.expert_sums[e]
            self.expert_sums[e] = upd if cur is None else cur + upd
            pending.append(self.expert_sums[e])
            if len(pending) >= 32:
                mx.eval(*pending)
                pending.clear()
        mx.eval(self.total, *pending)
        self.counts += counts

    # -- views ------------------------------------------------------------
    @property
    def count(self) -> mx.array:
        return mx.array(self.counts.astype(np.float32))

    @property
    def sum_xxT(self) -> mx.array:
        """Stacked sums (small models / tests only; experts stack ``[E, in, in]``)."""
        if not self.batch:
            return self.total
        return self.expert_chunk(0, self.batch)

    def expert_chunk(self, e0: int, e1: int) -> mx.array:
        z = mx.zeros((self.in_dim, self.in_dim), dtype=mx.float32)
        return mx.stack([z if s is None else s for s in self.expert_sums[e0:e1]])

    def layer_mean(self) -> mx.array:
        """Per-token ``E[x xᵀ]`` over everything this accumulator saw."""
        n = float(self.counts.sum()) if self.batch else float(self.counts)
        return self.total / max(n, 1.0)

    def mean(self) -> mx.array:
        """Dense: ``E[x xᵀ]``. Experts: per-expert means, stacked."""
        if not self.batch:
            return self.layer_mean()
        c = mx.maximum(self.count, 1.0)
        return self.sum_xxT / c[:, None, None]


def shrink_expert_hessians(
    sum_xxT: mx.array,
    counts: mx.array,
    prior_tokens: float = 256.0,
    h_bar: mx.array | None = None,
) -> mx.array:
    """Blend each expert's Hessian toward the layer-wide average.

    ``H_e = (S_e + λ · H̄) / (n_e + λ)`` where ``S_e = Σ x xᵀ`` over the
    expert's tokens, ``H̄`` the per-token average over all routed tokens and
    ``λ = prior_tokens``. A well-covered expert (``n_e ≫ λ``) keeps its own
    statistics; an unseen expert gets the layer average instead of a zero
    matrix. ``h_bar`` defaults to the average over the experts passed in; pass
    the full layer's average when shrinking a chunk of experts.
    """
    if h_bar is None:
        total = mx.maximum(counts.sum(), 1.0)
        h_bar = sum_xxT.sum(axis=0) / total
    return (sum_xxT + prior_tokens * h_bar[None]) / (counts + prior_tokens)[:, None, None]


# ---------------------------------------------------------------------------
# Affine group parameters (oQ-compatible)
# ---------------------------------------------------------------------------


def _round_to(x: mx.array, dtype) -> mx.array:
    return x.astype(dtype).astype(mx.float32)


def _group_params(wg: mx.array, imp: mx.array, bits: int, store_dtype):
    """Scale and bias for groups ``wg[..., G]`` with importance ``imp[..., G]``.

    Same candidates as oQe's clipping search (min/max anchored on the larger
    edge, scaled by the clip factors), picked by importance-weighted error,
    with scales and biases rounded to the storage dtype before scoring.
    """
    n_bins = float((1 << bits) - 1)
    eps = 1e-7
    w_min = mx.min(wg, axis=-1, keepdims=True)
    w_max = mx.max(wg, axis=-1, keepdims=True)

    best_s = best_b = best_err = None
    for edge, opposite, sign in ((w_max, w_min, -1.0), (w_min, w_max, 1.0)):
        raw = mx.maximum(mx.abs(edge - opposite) / n_bins, eps) * sign
        q0 = mx.round(edge / raw)
        scale0 = mx.where(q0 != 0, edge / q0, raw)
        bias0 = mx.where(q0 == 0, mx.zeros_like(edge), edge)
        for factor in _CLIP_FACTORS:
            s = _round_to(scale0 * factor, store_dtype)
            s = mx.where(mx.abs(s) < eps, mx.full(s.shape, eps * sign, mx.float32), s)
            b = _round_to(bias0, store_dtype)
            q = mx.clip(mx.round((wg - b) / s), 0, n_bins)
            err = mx.sum(imp * (wg - (q * s + b)) ** 2, axis=-1, keepdims=True)
            if best_err is None:
                best_s, best_b, best_err = s, b, err
            else:
                take = err < best_err
                best_err = mx.where(take, err, best_err)
                best_s = mx.where(take, s, best_s)
                best_b = mx.where(take, b, best_b)
    return best_s, best_b


def pack_codes(codes: mx.array, bits: int) -> mx.array:
    """Pack integer codes ``[..., in]`` exactly like ``mx.quantize`` (affine)."""
    lead = codes.shape[:-1]
    flat = codes.reshape(-1, codes.shape[-1]).astype(mx.uint32)
    if bits in (2, 4, 8):
        per = 32 // bits
        shifts = mx.power(mx.array(2, mx.uint32), mx.arange(0, 32, bits, dtype=mx.uint32))
        packed = mx.sum(flat.reshape(flat.shape[0], -1, per) * shifts, axis=2)
    else:
        bit_values = mx.bitwise_and(
            mx.right_shift(flat[..., None], mx.arange(bits, dtype=mx.uint32)), 1
        ).reshape(flat.shape[0], -1, 32)
        packed = mx.sum(mx.left_shift(bit_values, mx.arange(32, dtype=mx.uint32)), axis=-1)
    return packed.astype(mx.uint32).reshape(*lead, -1)


# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------


def _upper_inverse_cholesky(h: mx.array, damp: float, max_tries: int = 6) -> mx.array:
    """Upper Cholesky factor of ``H⁻¹`` with diagonal damping; retries on failure."""
    n = h.shape[-1]
    eye = mx.eye(n, dtype=mx.float32)
    diag = mx.diagonal(h, axis1=-2, axis2=-1)
    mean_diag = mx.maximum(mx.mean(diag, axis=-1, keepdims=True), 1e-8)
    d = damp
    for _ in range(max_tries):
        hd = h + (d * mean_diag)[..., None] * eye
        lower = mx.linalg.cholesky(hd, stream=mx.cpu)
        hinv = mx.linalg.cholesky_inv(lower, stream=mx.cpu)
        upper = mx.linalg.cholesky(hinv, upper=True, stream=mx.cpu)
        mx.eval(upper)
        if not bool(mx.any(mx.isnan(upper)).item()):
            return upper
        d *= 10.0
        logger.warning("oQ-T: Hessian not positive definite, raising damping to %g", d)
    raise RuntimeError("oQ-T: Hessian could not be made positive definite")


def _gptq_core(W, U, imp_all, bits, group_size, block_size, store_dtype, static=None):
    """One GPTQ sweep. Returns float ``(codes, scales, biases)``.

    ``static=None``: group parameters are searched when the sweep reaches each
    group, on the error-updated weights ("dynamic"). ``static=(scales,
    biases)``: fixed parameters chosen up front; the sweep only picks codes.
    ``W`` is consumed (updated in place).
    """
    in_dim = W.shape[-1]
    n_bins = float((1 << bits) - 1)
    n_groups = in_dim // group_size
    B, out_dim = W.shape[0], W.shape[1]
    codes = mx.zeros((B, out_dim, in_dim), dtype=mx.uint8)
    if static is None:
        scales = mx.zeros((B, out_dim, n_groups), dtype=mx.float32)
        biases = mx.zeros((B, out_dim, n_groups), dtype=mx.float32)
    else:
        scales, biases = (x.astype(mx.float32) for x in static)

    for i1 in range(0, in_dim, block_size):
        i2 = min(i1 + block_size, in_dim)
        W1 = W[:, :, i1:i2]
        U1 = U[:, i1:i2, i1:i2]
        E1 = mx.zeros_like(W1)
        Q1 = mx.zeros(W1.shape, dtype=mx.uint8)
        s = b = None
        for j in range(i2 - i1):
            col = i1 + j
            if col % group_size == 0:
                g = col // group_size
                if static is None:
                    imp = imp_all[:, None, col : col + group_size]
                    s, b = _group_params(W1[:, :, j : j + group_size], imp, bits, store_dtype)
                    scales[:, :, g] = s[..., 0]
                    biases[:, :, g] = b[..., 0]
                else:
                    s, b = scales[:, :, g : g + 1], biases[:, :, g : g + 1]
            wc = W1[:, :, j]
            q = mx.clip(mx.round((wc - b[..., 0]) / s[..., 0]), 0, n_bins)
            dq = q * s[..., 0] + b[..., 0]
            err = (wc - dq) / U1[:, j, j][:, None]
            Q1[:, :, j] = q.astype(mx.uint8)
            E1[:, :, j] = err
            if j + 1 < i2 - i1:
                W1[:, :, j + 1 :] = W1[:, :, j + 1 :] - err[:, :, None] * U1[:, j, j + 1 :][:, None, :]
            if j % 32 == 31:
                mx.eval(W1, E1, Q1, scales, biases)
        codes[:, :, i1:i2] = Q1
        if i2 < in_dim:
            W[:, :, i2:] = W[:, :, i2:] - E1 @ U[:, i1:i2, i2:]
        mx.eval(W, codes, scales, biases)
    return codes, scales, biases


def _dequant_float(codes, scales, biases, group_size):
    B, out_dim, in_dim = codes.shape
    g = codes.astype(mx.float32).reshape(B, out_dim, in_dim // group_size, group_size)
    return (g * scales[..., None] + biases[..., None]).reshape(B, out_dim, in_dim)


def _batched_output_error(w, wq, h):
    """Per-batch ``tr((W-Q) H (W-Q)ᵀ)``."""
    d = w - wq
    return mx.sum((d @ h) * d, axis=(-2, -1))


def refine_codes(W, H, codes, scales, biases, bits, group_size, *, sweeps=1, block_size=128):
    """M6: coordinate-descent refinement of integer codes (scales/biases fixed).

    Minimizes the exact layer output error ``tr((Q-W) H (Q-W)ᵀ)``. Rows are
    independent, so every row (and every expert) is refined in parallel; within
    a row, columns are visited in order and each code moves by ±1 only when
    that strictly lowers the loss, so the result is never worse than the input.
    Updates to the gradient ``G = (Q-W) H`` are applied lazily per column block,
    like GPTQ, so one sweep costs about as much as one GPTQ pass.

    Shapes: ``W, codes [B, out, in]``, ``H [B, in, in]``, ``scales/biases
    [B, out, in/group_size]``. Returns refined codes (uint8) and the number of
    codes changed.
    """
    B, out_dim, in_dim = W.shape
    n_bins = float((1 << bits) - 1)
    C = codes.astype(mx.float32)
    S = mx.repeat(scales, group_size, axis=-1)                 # [B, out, in]
    Bi = mx.repeat(biases, group_size, axis=-1)
    E = C * S + Bi - W                                         # Q - W
    G = E @ H                                                  # [B, out, in]
    Hd = mx.diagonal(H, axis1=-2, axis2=-1)                    # [B, in]
    changed = 0
    for _ in range(max(0, int(sweeps))):
        for i1 in range(0, in_dim, block_size):
            i2 = min(i1 + block_size, in_dim)
            Gb = G[:, :, i1:i2]
            Hb = H[:, i1:i2, i1:i2]
            D = mx.zeros((B, out_dim, i2 - i1), dtype=mx.float32)
            for j in range(i2 - i1):
                col = i1 + j
                s = S[:, :, col]
                g = Gb[:, :, j]
                hjj = Hd[:, col][:, None]
                c = C[:, :, col]
                gain_up = 2.0 * s * g + s * s * hjj             # loss change for +1
                gain_dn = -2.0 * s * g + s * s * hjj            # loss change for -1
                up_ok = (c + 1 <= n_bins) & (gain_up < -1e-12)
                dn_ok = (c - 1 >= 0) & (gain_dn < -1e-12)
                take_up = up_ok & (~dn_ok | (gain_up <= gain_dn))
                take_dn = dn_ok & ~take_up
                delta = mx.where(take_up, 1.0, mx.where(take_dn, -1.0, 0.0))
                step = delta * s                                # change of Q[:, :, col]
                C[:, :, col] = c + delta
                D[:, :, j] = step
                Gb = Gb + step[:, :, None] * Hb[:, j, :][:, None, :]
                if j % 32 == 31:
                    mx.eval(C, D, Gb)
            # Propagate this block's changes to the gradient outside the block.
            G[:, :, i1:i2] = Gb
            if i1 > 0:
                G[:, :, :i1] = G[:, :, :i1] + D @ H[:, i1:i2, :i1]
            if i2 < in_dim:
                G[:, :, i2:] = G[:, :, i2:] + D @ H[:, i1:i2, i2:]
            changed += int(mx.sum(D != 0).item())
            mx.eval(C, G)
    return C.astype(mx.uint8), changed


def gptq_quantize(
    w: mx.array,
    h: mx.array,
    bits: int,
    group_size: int = 64,
    *,
    damp: float = 0.01,
    block_size: int = 128,
    store_dtype=mx.bfloat16,
    group_params: str = "best",
    refine_sweeps: int = 1,
):
    """Quantize ``w[..., out, in]`` against Hessian ``h[..., in, in]`` with GPTQ.

    ``h`` is ``E[x xᵀ]`` of the layer input (any positive scale works); a 2-D
    ``h`` is shared across a batched ``w``. Returns ``(packed, scales,
    biases)`` in ``mx.quantize`` affine layout with scales/biases in
    ``store_dtype``.

    ``group_params``:

    * ``"static"`` — oQe's own clipping search + least-squares refit picks
      each group's scale/bias up front (Hessian-diagonal weighted); GPTQ only
      picks codes. With nothing to compensate it reproduces oQe exactly.
    * ``"dynamic"`` — parameters are searched per group on the
      error-updated weights; stronger when inputs are correlated.
    * ``"best"`` (default) — run both and keep, per batch element (expert),
      whichever has the lower output error ``tr((W-Q) H (W-Q)ᵀ)``. Never worse
      than either, at twice the cost.
    """
    if bits not in _VALID_BITS:
        raise ValueError(f"bits must be one of {_VALID_BITS}, got {bits}")
    if group_params not in ("static", "dynamic", "best"):
        raise ValueError(f"group_params must be static, dynamic or best, got {group_params!r}")
    in_dim = w.shape[-1]
    if in_dim % group_size:
        raise ValueError(f"in_dim {in_dim} is not a multiple of group_size {group_size}")
    block_size = max(group_size, (block_size // group_size) * group_size)

    batched = w.ndim == 3
    W0 = w.astype(mx.float32)
    if not batched:
        W0 = W0[None]
    H = h.astype(mx.float32)
    if H.ndim == 2:
        H = mx.broadcast_to(H[None], (W0.shape[0], in_dim, in_dim))
    H = mx.array(H)

    # Dead inputs (never active): neutral Hessian row, weight column zeroed.
    diag = mx.diagonal(H, axis1=-2, axis2=-1)                  # [B, in]
    dead = diag <= 0
    H_err = H                                                  # selection metric, undamped
    H = H + mx.where(dead, 1.0, 0.0)[..., None] * mx.eye(in_dim, dtype=mx.float32)
    W0 = mx.where(dead[:, None, :], 0.0, W0)
    imp_all = mx.maximum(mx.where(dead, 0.0, diag), 1e-8)       # group-search weights
    U = _upper_inverse_cholesky(H, damp)                       # [B, in, in]

    candidates = []
    if group_params in ("static", "best"):
        from .oq import _weighted_affine_quantize  # lazy: oq imports this module lazily too

        _, s0, b0 = _weighted_affine_quantize(W0.astype(store_dtype), group_size, bits, imp_all)
        candidates.append(_gptq_core(mx.array(W0), U, imp_all, bits, group_size,
                                     block_size, store_dtype, static=(s0, b0)))
    if group_params in ("dynamic", "best"):
        candidates.append(_gptq_core(mx.array(W0), U, imp_all, bits, group_size,
                                     block_size, store_dtype))

    if refine_sweeps > 0:
        # M6: polish every candidate against the exact (undamped) objective
        # before choosing, so "best" stays never worse than either variant.
        refined = []
        for c, sc, bi in candidates:
            c2, n_changed = refine_codes(W0, H_err, c, sc, bi, bits, group_size,
                                         sweeps=refine_sweeps, block_size=block_size)
            logger.debug("oQ-T: refinement changed %d codes", n_changed)
            refined.append((c2, sc, bi))
        candidates = refined

    codes, scales, biases = candidates[0]
    if len(candidates) == 2:
        errs = [_batched_output_error(W0, _dequant_float(c, s, b, group_size), H_err)
                for c, s, b in candidates]
        take = (errs[1] < errs[0])                              # [B]: dynamic wins
        codes = mx.where(take[:, None, None], candidates[1][0], codes)
        scales = mx.where(take[:, None, None], candidates[1][1], scales)
        biases = mx.where(take[:, None, None], candidates[1][2], biases)
        logger.debug("oQ-T: dynamic group params won %d/%d", int(take.sum().item()), take.size)

    packed = pack_codes(codes, bits)
    scales = scales.astype(store_dtype)
    biases = biases.astype(store_dtype)
    if not batched:
        packed, scales, biases = packed[0], scales[0], biases[0]
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def gptq_quantize_experts(
    w: mx.array,
    acc: "HessianAccumulator",
    bits: int,
    group_size: int = 64,
    *,
    prior_tokens: float = 256.0,
    expert_chunk: int = 32,
    **kwargs,
):
    """GPTQ for stacked experts ``w[E, out, in]`` with per-expert statistics.

    Hessians come from ``acc`` (a batched :class:`HessianAccumulator`), are
    shrunk toward the layer average (:func:`shrink_expert_hessians`) and are
    processed ``expert_chunk`` experts at a time to bound memory
    (``chunk · in² · 4`` bytes per factor).
    """
    h_bar = acc.layer_mean()
    counts = acc.count
    parts = []
    for e0 in range(0, w.shape[0], expert_chunk):
        e1 = min(e0 + expert_chunk, w.shape[0])
        h = shrink_expert_hessians(acc.expert_chunk(e0, e1), counts[e0:e1], prior_tokens, h_bar)
        parts.append(gptq_quantize(w[e0:e1], h, bits, group_size, **kwargs))
        del h
        mx.clear_cache()
    packed = mx.concatenate([p[0] for p in parts], axis=0)
    scales = mx.concatenate([p[1] for p in parts], axis=0)
    biases = mx.concatenate([p[2] for p in parts], axis=0)
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def output_error(w: mx.array, wq: mx.array, h: mx.array) -> float:
    """Relative output error ``tr((W-Q) H (W-Q)ᵀ) / tr(W H Wᵀ)`` (batched ok)."""
    d = (w - wq).astype(mx.float32)
    w = w.astype(mx.float32)
    num = mx.sum((d @ h) * d)
    den = mx.sum((w @ h) * w)
    return float((num / mx.maximum(den, 1e-12)).item())


# ===========================================================================
# Milestone 2 — the streamed sweep
# ===========================================================================
#
# One extra pass over the checkpoint, after oQ's imatrix/sensitivity pass and
# quant plan, with one decoder block resident at a time:
#
#   for each block:
#     1. forward the calibration hidden states through the float block while
#        a collector records the full input Hessian of every Linear and, per
#        expert, of every SwitchLinear
#     2. quantize each tensor with GPTQ at the bits/group size the oQ plan
#        assigned it (``_get_predicate_bits``, the same call the writer makes)
#     3. swap the dequantized weights into the block and forward again: the
#        *quantized* block's output becomes the next block's input, so every
#        block compensates the error accumulated upstream
#     4. save the block's results to the cache (resumable)
#
# The writer then takes a tensor from the cache when name, bits, group size
# and shape all match, and falls back to oQe otherwise (embeddings, lm_head,
# MTP, vision, multi-linear MLA weights, pre-quantized source tensors).

OQT_VERSION = "oqt-sweep-v1"


def _oq():
    # Attribute access at call time so tests can monkeypatch oq helpers.
    from . import oq

    return oq


class OQTHessianCollector:
    """Hessian-collecting twin of oQ's imatrix collector.

    Reuses ``OQImatrixCollector.install/restore`` (same capture predicate,
    same name prefixes, same skip set) but records ``Σ x xᵀ`` instead of the
    diagonal. Projections that receive the *same* input array in the same
    forward (q/k/v, gate/up) share one accumulator.
    """

    def __init__(self, skip: frozenset = frozenset()):
        base = _oq().OQImatrixCollector(skip=skip)
        self._base = base
        # Route the base collector's capture callbacks here.
        base.collect_dense = self.collect_dense
        base.collect_switch = self.collect_switch
        base.collect_multi = self.collect_multi
        base.collect_glm5_next_embed_q = lambda *a, **k: None
        self.hessians: dict[str, HessianAccumulator] = {}
        self.alias: dict[str, str] = {}
        self.unsupported: set[str] = set()
        self._seen_inputs: dict[int, tuple[Any, str]] = {}

    def install(self, block, name_prefix: str) -> dict[str, Any]:
        n = self._base.install(block, name_prefix=name_prefix)
        modules = dict(self._base._original_modules)
        logger.debug("oQ-T: installed %d capture modules", n)
        return modules

    def restore(self, block) -> None:
        self._base.restore(block)

    def end_forward(self) -> None:
        self._seen_inputs.clear()

    def _shared(self, name: str, x, batch: int) -> bool:
        """True when ``x`` was already accumulated for a compatible module this forward.

        Compatible means same input width *and* same kind: dense with dense,
        experts with the same expert count (a shared expert and the routed
        experts can receive the very same hidden-state array).
        """
        prev = self._seen_inputs.get(id(x))
        if prev is not None and prev[0] is x:
            owner = prev[1]
            acc = self.hessians.get(owner)
            if (owner != name and acc is not None and acc.batch == batch
                    and acc.in_dim == int(x.shape[-1])):
                self.alias.setdefault(name, owner)
                return True
        self._seen_inputs[id(x)] = (x, name)
        return False

    def collect_dense(self, name: str, module, x) -> None:
        in_dim = self._base._module_in_dim(module)
        if not getattr(x, "shape", ()) or int(x.shape[-1]) != in_dim:
            return
        if name in self.alias or self._shared(name, x, 0):
            return
        acc = self.hessians.get(name)
        if acc is None:
            acc = self.hessians[name] = HessianAccumulator(in_dim)
        acc.add(x)

    def collect_switch(self, name: str, module, x, indices) -> None:
        weight_shape = getattr(module.weight, "shape", ())
        n_experts = int(getattr(module, "num_experts", weight_shape[0]))
        in_dim = int(getattr(module, "input_dims", weight_shape[-1]))
        if not getattr(x, "shape", ()) or int(x.shape[-1]) != in_dim:
            return
        if name in self.alias or self._shared(name, x, n_experts):
            return
        acc = self.hessians.get(name)
        if acc is None:
            acc = self.hessians[name] = HessianAccumulator(in_dim, batch=n_experts)
        xf = x.reshape(-1, in_dim)
        idx = np.asarray(indices)
        rows = idx.reshape(-1, idx.shape[-1]) if idx.ndim > 1 else idx.reshape(-1, 1)
        if xf.shape[0] == rows.shape[0]:
            acc.add_routed(xf, mx.array(rows))         # token-level input (gate/up)
        elif xf.shape[0] == rows.size:
            acc.add_pairs(xf, mx.array(rows.reshape(-1)))  # pair-level input (down)
        else:
            self.unsupported.add(name)

    def collect_multi(self, name: str, module, x, *, transpose: bool = True) -> None:
        # MLA MultiLinear weights stay on the oQe path for now.
        self.unsupported.add(name)

    def accumulator(self, name: str) -> HessianAccumulator | None:
        return self.hessians.get(self.alias.get(name, name))


def _resolve_module(root, dotted: str):
    m = root
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() and isinstance(m, (list, tuple)) else getattr(m, part)
    return m


def _file_digest(path: str | None) -> str:
    """Content hash of a calibration path (file or directory of .txt files)."""
    if not path:
        return ""
    p = Path(path).expanduser()
    h = hashlib.sha256()
    if p.is_file():
        h.update(p.read_bytes())
    elif p.is_dir():
        for f in sorted(p.rglob("*.txt")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    else:
        h.update(str(path).encode())     # built-in dataset name
    return h.hexdigest()


def _packed_shape(shape: tuple[int, ...], bits: int) -> tuple[int, ...]:
    return (*shape[:-1], shape[-1] * bits // 32)


class OQTCache:
    """Per-layer results on disk: ``layer-NNN.safetensors`` + ``layer-NNN.json``.

    Each layer's JSON carries a chain digest (this layer's settings plus the
    previous layer's digest), because a block's inputs depend on every
    quantized block before it. A layer is reused only when its digest matches.
    """

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, layer_idx: int):
        stem = self.dir / f"layer-{layer_idx:03d}"
        return stem.with_suffix(".safetensors"), stem.with_suffix(".json")

    def load(self, layer_idx: int, digest: str):
        tensors_path, meta_path = self._paths(layer_idx)
        if not (tensors_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("digest") != digest:
            return None
        return meta

    def save(self, layer_idx: int, digest: str, tensors: dict, meta: dict) -> None:
        tensors_path, meta_path = self._paths(layer_idx)
        tmp = tensors_path.with_name(tensors_path.stem + ".tmp.safetensors")
        if tensors:
            mx.save_safetensors(str(tmp), tensors)
            tmp.replace(tensors_path)
        else:
            mx.save_safetensors(str(tensors_path), {"_empty": mx.zeros((1,))})
        meta = {**meta, "digest": digest}
        meta_path.write_text(json.dumps(meta, indent=1))


class OQTResults:
    """Writer-side lookup over a finished sweep's cache."""

    def __init__(self, cache: OQTCache, layers: dict[int, dict]):
        self.cache = cache
        self.layers = layers
        self.index: dict[str, tuple[int, dict]] = {}
        for layer_idx, meta in layers.items():
            for base, info in meta.get("tensors", {}).items():
                self.index[base] = (layer_idx, info)
        self._loaded: tuple[int, dict] | None = None
        self.used: list[str] = []
        self.declined: list[dict] = []

    def _tensors(self, layer_idx: int) -> dict:
        if self._loaded is None or self._loaded[0] != layer_idx:
            self._loaded = (layer_idx, mx.load(str(self.cache._paths(layer_idx)[0])))
        return self._loaded[1]

    def lookup(self, tensor_name: str, bits: int, group_size: int, shape):
        if not tensor_name.endswith(".weight"):
            return None
        base = tensor_name[: -len(".weight")]
        hit = self.index.get(base)
        if hit is None:
            return None
        layer_idx, info = hit
        cached_bits = int(info["bits"])
        # A routing-protection upgrade (M4) deliberately differs from the plan;
        # anything else that differs means the plan changed since the sweep.
        planned_ok = cached_bits == bits or (info.get("bits_override")
                                             and info.get("planned_bits") == bits)
        if not planned_ok or info["group_size"] != group_size:
            self.declined.append({"tensor": base, "reason": "plan changed",
                                  "cached": [cached_bits, info["group_size"]],
                                  "writer": [bits, group_size]})
            return None
        t = self._tensors(layer_idx)
        qw, scales, biases = t[f"{base}.weight"], t[f"{base}.scales"], t[f"{base}.biases"]
        if tuple(qw.shape) != _packed_shape(tuple(shape), cached_bits):
            self.declined.append({"tensor": base, "reason": "shape mismatch",
                                  "cached": list(qw.shape), "writer": list(shape)})
            return None
        self.used.append(base)
        return qw, scales, biases, cached_bits

    def report(self) -> dict:
        per_tensor = {}
        for meta in self.layers.values():
            per_tensor.update(meta.get("tensors", {}))
        errs = [(v["err_oqt"], v["err_oqe"]) for v in per_tensor.values()
                if v.get("err_oqt") is not None and v.get("err_oqe")]
        summary = {}
        if errs:
            ratio = np.array([a / b for a, b in errs if b > 0])
            summary = {
                "tensors_compared": len(errs),
                "median_err_ratio_oqt_vs_oqe": float(np.median(ratio)),
                "mean_err_ratio_oqt_vs_oqe": float(np.mean(ratio)),
                "share_tensors_improved": float(np.mean(ratio < 1.0)),
            }
        routes = [m["routing"] for m in self.layers.values() if m.get("routing")]
        if routes:
            summary["routing_mass_shift_median"] = float(np.median([r["mass_shift"] for r in routes]))
            summary["routing_layers_upgraded"] = sum(1 for r in routes if r.get("upgraded"))
            summary["routing_tensors_upgraded"] = sum(len(r.get("upgraded") or {}) for r in routes)
        usage = [m["expert_usage"] for m in self.layers.values() if m.get("expert_usage")]
        if usage:
            top20 = float(np.median([u["top20pct_share"] for u in usage]))
            summary["expert_top20pct_share_median"] = top20
            # M5 decision aid: tiers pay off only when a minority of experts
            # carries most tokens, consistently across layers.
            summary["expert_tiering_signal"] = (
                "strong" if top20 >= 0.6 else "moderate" if top20 >= 0.45 else "weak")
        cal = getattr(self, "calibration", None)
        if cal:
            summary["calibration_windows"] = cal
        return {
            "version": OQT_VERSION,
            "summary": summary,
            "used": sorted(set(self.used)),
            "declined_by_writer": self.declined,
            "skipped_in_sweep": {k: v for m in self.layers.values()
                                 for k, v in m.get("skipped", {}).items()},
            "layers": {str(k): {kk: vv for kk, vv in v.items() if kk != "tensors"}
                       for k, v in sorted(self.layers.items())},
            "tensors": per_tensor,
        }


def _relative_output_error(w: mx.array, wq: mx.array, acc: HessianAccumulator,
                           prior_tokens: float, expert_chunk: int = 32) -> float:
    """Mean relative output error, per expert for switch weights."""
    w = w.astype(mx.float32)
    wq = wq.astype(mx.float32)
    if not acc.batch:
        h = acc.layer_mean()
        return output_error(w, wq, h)
    h_bar = acc.layer_mean()
    counts = acc.count
    num = den = 0.0
    for e0 in range(0, w.shape[0], expert_chunk):
        e1 = min(e0 + expert_chunk, w.shape[0])
        h = shrink_expert_hessians(acc.expert_chunk(e0, e1), counts[e0:e1], prior_tokens, h_bar)
        d = w[e0:e1] - wq[e0:e1]
        num += float(mx.sum((d @ h) * d).item())
        den += float(mx.sum((w[e0:e1] @ h) * w[e0:e1]).item())
    return num / max(den, 1e-12)


_BITS_LADDER = (2, 3, 4, 5, 6, 8)


def _bits_up(bits: int, steps: int) -> int:
    """Next valid MLX affine bit width ``steps`` rungs above ``bits`` (capped at 8)."""
    if steps <= 0:
        return bits
    higher = [b for b in _BITS_LADDER if b > bits]
    return higher[min(steps, len(higher)) - 1] if higher else bits


def _find_router(block):
    """The MoE router of a decoder block: a 2-D Linear named ``gate`` whose parent
    also holds the routed experts (``switch_mlp``/``experts``). None for dense blocks."""
    for name, module in block.named_modules():
        if not name or name.split(".")[-1] != "gate":
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = _resolve_module(block, parent_name) if parent_name else block
        has_experts = any(hasattr(parent, k) for k in ("switch_mlp", "experts"))
        w = getattr(module, "weight", None)
        if has_experts and w is not None and getattr(w, "ndim", 0) == 2:
            return (parent, "gate", module, parent_name + "." if parent_name else "")
    return None


class _RouterTap:
    """Records the router's top-k experts per token, for teacher and student forwards."""

    def __init__(self, k: int):
        self.k = int(k)
        self.teacher: dict[int, tuple] = {}
        self.student: dict[int, mx.array] = {}
        self.record_as = "teacher"
        self.slot = 0
        self._attached = None
        self.parent_prefix = ""

    def attach(self, router):
        parent, attr, original, parent_prefix = router
        self.parent_prefix = parent_prefix
        tap = self

        def wrapped_call(x, *a, **kw):
            logits = original(x, *a, **kw)
            tap._record(logits)
            return logits

        proxy = _CallProxy(original, wrapped_call)
        setattr(parent, attr, proxy)
        self._attached = (parent, attr, original)

    def detach(self):
        if self._attached is not None:
            parent, attr, original = self._attached
            # MLX modules are dicts: the proxy was stored as an instance attribute
            # shadowing the module entry, so remove it, then restore the entry.
            parent.__dict__.pop(attr, None)
            setattr(parent, attr, original)
            self._attached = None

    def reset_student(self):
        self.student = {}

    def _record(self, logits):
        lg = logits.astype(mx.float32).reshape(-1, logits.shape[-1])
        k = min(self.k, lg.shape[-1])
        idx = mx.argpartition(-lg, kth=k - 1, axis=-1)[:, :k]
        if self.record_as == "teacher":
            probs = mx.softmax(lg, axis=-1)
            p = mx.take_along_axis(probs, idx, axis=-1)
            p = p / mx.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
            mx.eval(idx, p)
            self.teacher[self.slot] = (idx, p)
        else:
            mx.eval(idx)
            self.student[self.slot] = idx

    def shift(self) -> tuple[float, float]:
        """(routing mass moved to other experts, share of tokens with any change)."""
        mass_sum = flip_sum = n = 0.0
        for slot, (t_idx, t_p) in self.teacher.items():
            s_idx = self.student.get(slot)
            if s_idx is None:
                continue
            kept = mx.any(t_idx[:, :, None] == s_idx[:, None, :], axis=-1)
            dropped = mx.sum(t_p * (1.0 - kept.astype(mx.float32)), axis=-1)
            mass_sum += float(mx.sum(dropped).item())
            flip_sum += float(mx.sum(mx.any(~kept, axis=-1).astype(mx.float32)).item())
            n += t_idx.shape[0]
        return (mass_sum / n, flip_sum / n) if n else (0.0, 0.0)


class _CallProxy:
    """Stands in for a module during a forward: calls go through ``fn``,
    attribute reads reach the original module."""

    def __init__(self, original, fn):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_fn", fn)

    def __call__(self, *a, **kw):
        return object.__getattribute__(self, "_fn")(*a, **kw)

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, "_original"), key)


def _expert_usage(collector) -> dict:
    """M5 measurement: how unevenly the block's tokens use its experts."""
    for acc in collector.hessians.values():
        if acc.batch and acc.counts.sum() > 0:
            c = np.sort(acc.counts.astype(np.float64))[::-1]
            p = c / c.sum()
            nz = p[p > 0]
            entropy = float(-(nz * np.log(nz)).sum() / np.log(len(p))) if len(p) > 1 else 1.0
            k10 = max(1, int(round(0.10 * len(p))))
            k20 = max(1, int(round(0.20 * len(p))))
            return {"experts": int(len(p)), "entropy_norm": round(entropy, 4),
                    "top10pct_share": round(float(p[:k10].sum()), 4),
                    "top20pct_share": round(float(p[:k20].sum()), 4),
                    "unseen": int((c == 0).sum())}
    return {}


def _pack_long_windows(tokenizer, path, length: int, count: int, seed: int):
    """Pack whole documents (one JSON object with "text" per line) into windows of
    exactly ``length`` tokens. Every window starts at a document's beginning;
    documents are appended until the window is full, the tail is cut, and the
    next window starts at the next document. Returns [n, length] or None."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line)["text"])
    docs.sort(key=lambda d: hashlib.sha256(f"{seed}|{d[:200]}|{len(d)}".encode()).hexdigest())
    windows, cur = [], []
    for doc in docs:
        ids = tokenizer.encode(doc)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        cur.extend(int(i) for i in ids)
        if len(cur) >= length:
            windows.append(cur[:length])
            cur = []
            if len(windows) >= count:
                break
    if not windows:
        return None
    return mx.array(windows, dtype=mx.int32)


def run_streaming_sweep(
    model_path: str,
    config: dict,
    oq_level: float,
    group_size: int,
    *,
    cache_dir: str,
    calib_dataset: str,
    num_samples: int = 128,
    seq_length: int = 2048,
    target_dtype=mx.bfloat16,
    trust_remote_code: bool = False,
    progress_callback: Callable | None = None,
    calib_data: mx.array | None = None,
    calib_seed: int | None = None,
    prior_tokens: float = 256.0,
    damp: float = 0.01,
    group_params: str = "best",
    expert_chunk: int = 32,
    compare_oqe: bool = True,
    refine_sweeps: int = 1,
    refine_expert_sweeps: int = 0,
    routing_protect: bool = True,
    routing_mass_threshold: float = 0.02,
    routing_max_extra_bits: int = 2,
    routing_max_layers_frac: float = 0.25,
    long_calib_path: str | None = None,
    long_seq_length: int = 32768,
    long_share: float = 0.25,
) -> OQTResults:
    """Run the oQ-T sweep and return the writer-side lookup.

    ``config`` must already carry the quant plan (``_oq_boost_map``), exactly
    as the writer will see it.
    """
    oq = _oq()
    model_type = oq._stream_source_model_type(config)
    if model_type == "deepseek_v41":
        raise NotImplementedError(
            "oQ-T streaming sweep does not support DeepSeek V4.1 yet "
            "(it has its own calibration path); use oQe for this model")
    if not oq._stream_calibration_supported(model_type):
        raise ValueError(f"oQ-T needs a streamable layout; {model_type!r} is not one")

    from mlx_lm.tokenizer_utils import load as load_tokenizer

    source = Path(model_path)
    cache = OQTCache(cache_dir)
    seed = oq._OQE_STREAM_CALIB_SEED if calib_seed is None else int(calib_seed)
    use_long = bool(long_calib_path) and long_share > 0 and Path(long_calib_path).exists()
    tokenizer = load_tokenizer(source) if (calib_data is None or use_long) else None
    if calib_data is None:
        mx.random.seed(seed)
        calib_data = oq._load_calibration_data(
            tokenizer, dataset=calib_dataset, num_samples=num_samples, seq_length=seq_length)
    if calib_data is None:
        raise RuntimeError(f"oQ-T: no calibration data from {calib_dataset!r}")
    calib_data = calib_data[:num_samples]
    short_len = int(calib_data.shape[1])

    # Long windows (whole documents packed from their start) take long_share of
    # the same token total; the short windows shrink to keep the total fixed.
    long_data = None
    if use_long:
        total_tokens = int(calib_data.shape[0]) * short_len
        n_long = int(round(long_share * total_tokens / long_seq_length))
        if n_long >= 1:
            long_data = _pack_long_windows(tokenizer, long_calib_path, long_seq_length, n_long, seed)
        if long_data is not None:
            n_short = max(1, int(round((total_tokens - long_data.shape[0] * long_seq_length) / short_len)))
            calib_data = calib_data[:n_short]
            logger.info("oQ-T: %d long windows x %d tokens + %d short windows x %d tokens",
                        long_data.shape[0], long_seq_length, n_short, short_len)
        else:
            logger.warning("oQ-T: long documents too few or too short for one %d-token window; "
                           "calibrating on short windows only", long_seq_length)
    n_samples = int(calib_data.shape[0])

    batch_plan = oq._oqe_calibration_batch_plan(
        config, requested_samples=n_samples, seq_length=short_len)
    micro = max(1, int(batch_plan["micro_batch_size"]))

    is_qwen4_exp = model_type == "qwen4_exp"
    is_glm5_next = model_type == "glm5_next"
    if is_qwen4_exp:
        oq._qwen4_exp_apply_compat_patch()
    embed_weight = oq._streamed_embed_weight(source, config)

    # Each stream (short windows, long windows) gets its own embedding, masks
    # and positions; the sweep then iterates a flat list of slots.
    working, slots = [], []

    def add_stream(data, micro_size, kind):
        embedded = oq._streamed_embed(embed_weight, data, config)
        mx.eval(embedded)
        n = int(data.shape[0])
        ranges = [(lo, min(lo + micro_size, n)) for lo in range(0, n, micro_size)]
        if is_qwen4_exp:
            embedded, layer_masks, pos = oq._streamed_qwen4_exp_state(config, data, embedded)
            mx.eval(embedded)
            for lo, hi in ranges:
                working.append(embedded[lo:hi])
                slots.append({"kind": kind, "layer_masks": layer_masks,
                              "state": oq._streamed_qwen4_exp_slot_state(data, pos, lo, hi)})
        elif is_glm5_next:
            embedded, m_ = oq._streamed_glm5_next_state(config, embedded)
            mx.eval(embedded)
            for lo, hi in ranges:
                working.append(embedded[lo:hi])
                slots.append({"kind": kind, "mask": m_, "glm5": True,
                              "state": {"kind": oq._GLM5_NEXT_LAYER_STATE_KIND}})
        elif model_type == "flint":
            for lo, hi in ranges:
                working.append(embedded[lo:hi])
                slots.append({"kind": kind, "mask": None,
                              "state": {"kind": oq._FLINT_LAYER_STATE_KIND}})
        else:
            _, lm, pos = oq._prepare_layer_inputs(
                oq._StreamingNoModel(), [None], data[:1], embedded[:1],
                noncausal=oq._noncausal_attention(config))
            for lo, hi in ranges:
                working.append(embedded[lo:hi])
                slots.append({"kind": kind, "mask": lm[0], "state": pos})

    add_stream(calib_data, micro, "short")
    if long_data is not None:
        add_stream(long_data, 1, "long")
    del embed_weight
    mx.clear_cache()

    settings = {
        "version": OQT_VERSION, "source": str(source.resolve()),
        "calib": _file_digest(calib_dataset), "calib_name": str(calib_dataset),
        "samples": n_samples, "seq_length": short_len, "seed": seed,
        "oq_level": float(oq_level), "group_size": int(group_size),
        "dtype": str(target_dtype), "prior_tokens": prior_tokens, "damp": damp,
        "group_params": group_params,
        "long": ([_file_digest(long_calib_path), int(long_seq_length), float(long_share),
                  int(long_data.shape[0])] if long_data is not None else None),
        "refine": [int(refine_sweeps), int(refine_expert_sweeps)],
        "routing": [bool(routing_protect), float(routing_mass_threshold),
                    int(routing_max_extra_bits), float(routing_max_layers_frac)],
    }
    prev_digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    text_config = config.get("text_config") or {}
    total_layers = int(text_config.get("num_hidden_layers") or config.get("num_hidden_layers") or 0)
    skip = config.get("_oq_imatrix_capture_skip", frozenset())
    layers_meta: dict[int, dict] = {}
    t_start = time.monotonic()

    top_k = int(text_config.get("num_experts_per_tok") or config.get("num_experts_per_tok")
                or text_config.get("moe_top_k") or config.get("moe_top_k") or 0)
    max_upgraded_layers = max(0, int(np.ceil(routing_max_layers_frac * max(total_layers, 1))))
    upgraded_layers = 0

    def forward_all(block, layer_idx, collector=None, tap=None, commit=True):
        """Forward every slot. commit=True stores each output in place of its input;
        commit=False returns the outputs; commit=None drops them (router tap only)."""
        outs = []
        for slot in range(len(working)):
            sl = slots[slot]
            if "layer_masks" in sl:
                layer_mask = sl["layer_masks"][layer_idx]
            elif sl.get("glm5") and bool(getattr(block, "is_linear", False)):
                layer_mask = None
            else:
                layer_mask = sl["mask"]
            state = sl["state"]
            if tap is not None:
                tap.slot = slot
            out, _ = oq._forward_layer_result(block, working[slot], layer_mask, state,
                                              layer_idx=layer_idx)
            if out is None:
                raise RuntimeError(f"oQ-T: layer {layer_idx} forward returned no output")
            mx.eval(out)
            if collector is not None:
                collector.end_forward()
            elif commit:
                working[slot] = out
            elif commit is False:
                outs.append(out)
        return outs

    spill_dir = Path(cache_dir).expanduser() / ".spill"

    def spill_inputs():
        """Save the layer inputs to disk so the check pass can overwrite them in place."""
        spill_dir.mkdir(parents=True, exist_ok=True)
        for slot, x in enumerate(working):
            mx.save_safetensors(str(spill_dir / f"slot-{slot:05d}.safetensors"), {"x": x})

    def restore_inputs():
        for slot in range(len(working)):
            working[slot] = None
        mx.clear_cache()
        for slot in range(len(working)):
            x = mx.load(str(spill_dir / f"slot-{slot:05d}.safetensors"))["x"]
            mx.eval(x)
            working[slot] = x

    def drop_spill():
        if spill_dir.exists():
            for p in spill_dir.glob("slot-*.safetensors"):
                p.unlink()

    for layer_idx, block, _is_moe in oq._iter_streamed_layer_blocks(
        source, config, trust_remote_code=trust_remote_code
    ):
        prefix = f"{oq._STREAM_TEXT_LAYER_PREFIX}{layer_idx}."
        t_layer = time.monotonic()

        # Planned bits for every capture module of this block.
        probe = OQTHessianCollector(skip=skip)
        modules = probe.install(block, prefix)
        probe.restore(block)
        plan = {}
        for rel, module in modules.items():
            full = prefix + rel
            bits, gs, mode = oq._get_predicate_bits(full + ".weight", config, oq_level, group_size)
            plan[rel] = (bits, gs, mode)
        digest = hashlib.sha256(json.dumps(
            [prev_digest, layer_idx, sorted((k, list(v)) for k, v in plan.items())],
            sort_keys=True, default=str).encode()).hexdigest()

        meta = cache.load(layer_idx, digest)
        if meta is not None:
            tensors = mx.load(str(cache._paths(layer_idx)[0]))
            for base, info in meta.get("tensors", {}).items():
                rel = base[len(prefix):]
                module = _resolve_module(block, rel)
                module.weight = mx.dequantize(
                    tensors[f"{base}.weight"], tensors[f"{base}.scales"], tensors[f"{base}.biases"],
                    group_size=info["group_size"], bits=info["bits"]).astype(module.weight.dtype)
            meta["reused"] = True
            if meta.get("routing", {}).get("upgraded"):
                upgraded_layers += 1
            logger.info("oQ-T: layer %d reused from cache", layer_idx)
            forward_all(block, layer_idx)
        else:
            router = _find_router(block) if (routing_protect and top_k > 0) else None
            tap = _RouterTap(top_k) if router is not None else None
            collector = OQTHessianCollector(skip=skip)
            collector.install(block, prefix)
            if tap is not None:
                tap.attach(router)
                tap.record_as = "teacher"
            try:
                forward_all(block, layer_idx, collector)
            finally:
                collector.restore(block)
                if tap is not None:
                    tap.detach()

            originals = {rel: getattr(m, "weight", None) for rel, m in modules.items()}
            tensors, info_by_base, skipped = {}, {}, {}
            # Only the modules feeding the router can be quantized again (routing
            # protection); every other module's source weight and Hessian are
            # dropped as soon as it is quantized.
            router_parent = router[3] if tap is not None else None
            requant = {rel for rel in modules
                       if router_parent is not None and not rel.startswith(router_parent)}
            acc_users: dict[int, int] = {}
            for rel in modules:
                a = collector.accumulator(prefix + rel)
                if a is not None:
                    acc_users[id(a)] = acc_users.get(id(a), 0) + (1 if rel not in requant else 1_000_000)

            def quantize_one(rel, bits):
                """Quantize one capture module at ``bits``; returns its info or a skip reason."""
                module = modules[rel]
                full = prefix + rel
                _, gs, mode = plan[rel]
                acc = collector.accumulator(full)
                w = originals[rel]
                if bits is None or mode != "affine":
                    return None, f"plan: bits={bits} mode={mode}"
                if acc is None:
                    return None, "no activations captured"
                if full in collector.unsupported:
                    return None, "unsupported module kind"
                if w is None or not mx.issubdtype(w.dtype, mx.floating):
                    return None, "pre-quantized source tensor"
                if w.shape[-1] % gs:
                    return None, "in_dim not a multiple of group size"
                wf = w.astype(target_dtype)
                kw = dict(damp=damp, store_dtype=target_dtype, group_params=group_params)
                if acc.batch:
                    qw, sc, bi = gptq_quantize_experts(
                        wf, acc, bits, gs, prior_tokens=prior_tokens, expert_chunk=expert_chunk,
                        refine_sweeps=refine_expert_sweeps, **kw)
                else:
                    qw, sc, bi = gptq_quantize(wf, acc.layer_mean(), bits, gs,
                                               refine_sweeps=refine_sweeps, **kw)
                deq = mx.dequantize(qw, sc, bi, group_size=gs, bits=bits)
                info = {"bits": int(bits), "group_size": int(gs), "shape": list(w.shape),
                        "tokens": float(acc.counts.sum()) if acc.batch else float(acc.counts)}
                if acc.batch:
                    c = acc.counts
                    info["expert_tokens_p05"] = float(np.percentile(c, 5))
                    info["experts_unseen"] = int((c == 0).sum())
                if compare_oqe:
                    info["err_oqt"] = _relative_output_error(wf, deq, acc, prior_tokens, expert_chunk)
                    imp = (mx.diagonal(acc.layer_mean()) if not acc.batch else
                           mx.stack([mx.diagonal(s_) if s_ is not None else mx.zeros((acc.in_dim,))
                                     for s_ in acc.expert_sums]) / mx.maximum(acc.count, 1.0)[:, None])
                    oqp, oqs, oqb = oq._quantize_chunked(wf, gs, bits, "affine", importance=imp)
                    oqe_deq = mx.dequantize(oqp, oqs, oqb, group_size=gs, bits=bits)
                    info["err_oqe"] = _relative_output_error(wf, oqe_deq, acc, prior_tokens, expert_chunk)
                    del oqp, oqs, oqb, oqe_deq
                tensors[f"{full}.weight"], tensors[f"{full}.scales"], tensors[f"{full}.biases"] = qw, sc, bi
                info_by_base[full] = info
                module.weight = deq.astype(w.dtype)
                mx.eval(module.weight)
                mx.clear_cache()
                return info, None

            for rel in modules:
                _, reason = quantize_one(rel, plan[rel][0])
                if reason:
                    skipped[prefix + rel] = reason
                if rel in requant:
                    continue
                originals[rel] = None
                a = collector.accumulator(prefix + rel)
                if a is not None:
                    acc_users[id(a)] -= 1
                    if acc_users[id(a)] == 0 and a.batch:
                        a.expert_sums = [None] * a.batch   # counts stay for the usage stats
                        a.total = None
                mx.clear_cache()

            # Check the quantized block; its outputs replace the inputs in place.
            # A routing upgrade needs the inputs again, so they go to disk first.
            pre_router = [rel for rel in modules if rel in requant and prefix + rel in info_by_base]
            may_upgrade = (tap is not None and bool(pre_router) and routing_max_extra_bits > 0
                           and upgraded_layers < max_upgraded_layers)
            if may_upgrade:
                spill_inputs()
            routing = {}
            if tap is not None:
                tap.attach(router)
                tap.record_as = "student"
                tap.reset_student()
            try:
                forward_all(block, layer_idx, commit=True)
            finally:
                if tap is not None:
                    tap.detach()

            if tap is not None and tap.teacher:
                mass, flip = tap.shift()
                routing = {"mass_shift": round(mass, 5), "flip_rate": round(flip, 5),
                           "threshold": routing_mass_threshold}
                if may_upgrade and mass > routing_mass_threshold:
                    restore_inputs()
                    base_bits = {rel: info_by_base[prefix + rel]["bits"] for rel in pre_router}
                    best = (mass, dict(base_bits))
                    for extra in range(1, routing_max_extra_bits + 1):
                        new_bits = {rel: _bits_up(base_bits[rel], extra) for rel in pre_router}
                        for rel in pre_router:
                            quantize_one(rel, new_bits[rel])
                        tap.attach(router)
                        tap.reset_student()
                        try:
                            forward_all(block, layer_idx, commit=None)
                        finally:
                            tap.detach()
                        m2, f2 = tap.shift()
                        if m2 < best[0]:
                            best = (m2, new_bits)
                        if m2 <= routing_mass_threshold:
                            break
                    # Re-apply the best configuration and forward it once more:
                    # the same weights give the same outputs the trial produced.
                    final_bits = best[1]
                    for rel in pre_router:
                        quantize_one(rel, final_bits[rel])
                        if final_bits[rel] != base_bits[rel]:
                            info_by_base[prefix + rel]["bits_override"] = True
                            info_by_base[prefix + rel]["planned_bits"] = base_bits[rel]
                    forward_all(block, layer_idx, commit=True)
                    upgraded = {rel: [base_bits[rel], final_bits[rel]]
                                for rel in pre_router if final_bits[rel] != base_bits[rel]}
                    if upgraded:
                        upgraded_layers += 1
                    routing.update({"mass_shift_after": round(best[0], 5), "upgraded": upgraded})
            if may_upgrade:
                drop_spill()

            usage = _expert_usage(collector)
            del collector
            meta = {"layer": layer_idx, "tensors": info_by_base, "skipped": skipped,
                    "seconds": round(time.monotonic() - t_layer, 1)}
            if routing:
                meta["routing"] = routing
            if usage:
                meta["expert_usage"] = usage
            cache.save(layer_idx, digest, tensors, meta)
            del tensors

        layers_meta[layer_idx] = meta
        prev_digest = digest
        # Drop every reference to this layer (its modules, original weights,
        # check outputs, cached tensors, router tap) so clear_cache frees it
        # before the next layer is sourced; otherwise two layers overlap.
        block = modules = probe = originals = outs = tensors = None
        tap = router = quantize_one = None
        mx.synchronize()
        mx.clear_cache()
        done = layer_idx + 1
        msg = (f"oQ-T sweep: layer {done}/{total_layers or '?'} "
               f"({time.monotonic() - t_start:.0f}s elapsed, "
               f"active {mx.get_active_memory() / 1e9:.1f} GB, "
               f"peak {mx.get_peak_memory() / 1e9:.1f} GB, "
               f"cache {mx.get_cache_memory() / 1e9:.1f} GB)")
        r = meta.get("routing") or {}
        if r:
            msg += f", routing shift {r.get('mass_shift', 0):.3f}"
            if r.get("upgraded"):
                msg += f" -> {r.get('mass_shift_after', 0):.3f} after +bits on {len(r['upgraded'])} tensors"
        logger.info(msg)
        if progress_callback is not None:
            frac = done / total_layers if total_layers else 1.0
            try:
                progress_callback("oqt", 18.0 + 2.0 * frac, msg)
            except TypeError:
                progress_callback("oqt", 18.0 + 2.0 * frac)

    results = OQTResults(cache, layers_meta)
    results.calibration = {
        "short_windows": int(calib_data.shape[0]), "short_len": short_len,
        "long_windows": int(long_data.shape[0]) if long_data is not None else 0,
        "long_len": int(long_seq_length) if long_data is not None else 0}
    return results


# ===========================================================================
# Writer resume
# ===========================================================================


class WriterJournal:
    """Progress journal for oQ's tensor-by-tensor writer.

    Written after every output shard: how many tensors (in the writer's
    deterministic order) are fully inside written shards, the shard map, the
    per-tensor quantization config, and report lists. A rerun with the same
    inputs skips those tensors; a rerun with different inputs deletes the
    stale shards and starts over. Removed when the build completes.
    """

    NAME = ".oq_writer_journal.json"

    def __init__(self, output, digest: str):
        self.output = Path(output)
        self.path = self.output / self.NAME
        self.digest = digest

    def restore(self):
        """Returns (tensors_done, shards_written, weight_map, per_layer_config, extra)."""
        if not self.path.exists():
            return 0, 0, {}, {}, {}
        try:
            state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            state = {}
        shards = sorted(set((state.get("weight_map") or {}).values()))
        if state.get("digest") != self.digest or not all((self.output / s_).exists() for s_ in shards):
            for s_ in shards:
                (self.output / s_).unlink(missing_ok=True)
            self.path.unlink(missing_ok=True)
            logger.info("oQ writer: previous partial output does not match these inputs; starting over")
            return 0, 0, {}, {}, {}
        logger.info("oQ writer: resuming after %d tensors (%d shards already written)",
                    state["tensors_done"], state["shards"])
        return (int(state["tensors_done"]), int(state["shards"]), dict(state["weight_map"]),
                dict(state["per_layer_config"]), dict(state.get("extra") or {}))

    def record(self, tensors_done, shards, weight_map, per_layer_config, extra=None):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "digest": self.digest, "tensors_done": int(tensors_done), "shards": int(shards),
            "weight_map": weight_map, "per_layer_config": per_layer_config,
            "extra": extra or {}}, default=str))
        tmp.replace(self.path)

    def finish(self):
        self.path.unlink(missing_ok=True)


def writer_digest(**parts) -> str:
    """Identity of everything that decides the writer's output."""
    norm = {}
    for k, v in parts.items():
        if isinstance(v, OQTResults):
            v = sorted((b, i["bits"], i["group_size"]) for b, (_, i) in v.index.items())
        elif isinstance(v, (str, Path)) and k.endswith("_path") and v and Path(v).exists():
            st = Path(v).stat()
            v = [str(v), st.st_size, int(st.st_mtime)]
        norm[k] = v
    return hashlib.sha256(json.dumps(norm, sort_keys=True, default=str).encode()).hexdigest()


# ===========================================================================
# M3 — scoring quants against the original model
# ===========================================================================
#
# record_reference streams the BF16 source one block at a time over fixed
# evaluation windows and stores, per token, the original model's top-k next
# token log-probabilities plus the remaining mass, and per MoE layer the
# router's chosen experts and weights. score_against_reference runs a loaded
# quant over the same windows and reports KL divergence, top-1 agreement and
# routing flips, per slice and per layer.

REFERENCE_VERSION = "oqt-ref-v1"


def _streamed_head(source: Path, config: dict):
    """The model's final head as a callable hidden -> logits, from source tensors."""
    oq = _oq()
    import mlx.nn as nn

    plan = oq._streamed_source_plan(source, config)
    text_config = config.get("text_config") or config
    model_type = oq._stream_source_model_type(config)
    if model_type == "qwen4_exp":
        from mlx_vlm.models.qwen4_exp.config import TextConfig
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual

        prefix = "language_model.model.hyper_connection_mixer."
        mixer = Qwen4ExpGatedResidual(TextConfig.from_dict(text_config), use_combine=False)
        mixer.load_weights([(k[len(prefix):], plan.pop(k)) for k in list(plan) if k.startswith(prefix)],
                           strict=True)
        mixer.eval()
        pre = mixer
    else:
        norm_w = plan.pop("language_model.model.norm.weight")
        pre = nn.RMSNorm(int(norm_w.shape[0]), eps=float(text_config.get("rms_norm_eps") or 1e-6))
        pre.weight = norm_w
    head_key = "language_model.lm_head.weight"
    head_w = plan.pop(head_key) if head_key in plan else plan.pop(oq._STREAM_EMBED_KEY)
    head_w = head_w[:] if hasattr(head_w, "__getitem__") and not isinstance(head_w, mx.array) else head_w
    mx.eval(head_w)

    def head(hidden):
        return pre(hidden) @ head_w.T

    return head


def _topk_logprobs(logits: mx.array, k: int):
    """Top-k log-probabilities and the log of the remaining probability mass."""
    logp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    idx = mx.argpartition(-logp, kth=k - 1, axis=-1)[..., :k]
    top = mx.take_along_axis(logp, idx, axis=-1)
    order = mx.argsort(-top, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    top = mx.take_along_axis(top, order, axis=-1)
    rest = mx.log(mx.maximum(1.0 - mx.exp(top).sum(axis=-1), 1e-12))
    return idx, top, rest


def record_reference(model_path: str, config: dict, windows: mx.array, out_path: str, *,
                     slices: list[str] | None = None, top_k: int = 64,
                     trust_remote_code: bool = False, token_chunk: int = 256,
                     progress: Callable | None = None) -> dict:
    """Stream the source model over ``windows`` [N, L] and save the reference."""
    oq = _oq()
    model_type = oq._stream_source_model_type(config)
    if model_type == "deepseek_v41":
        raise NotImplementedError("M3 reference does not support DeepSeek V4.1 yet")
    if model_type not in ("qwen4_exp",) and not oq._stream_calibration_supported(model_type):
        raise ValueError(f"M3 needs a streamable layout; {model_type!r} is not one")
    source = Path(model_path)
    if model_type == "qwen4_exp":
        oq._qwen4_exp_apply_compat_patch()
    text_config = config.get("text_config") or {}
    router_k = int(text_config.get("num_experts_per_tok") or config.get("num_experts_per_tok") or 0)

    embed_weight = oq._streamed_embed_weight(source, config)
    embedded = oq._streamed_embed(embed_weight, windows, config)
    mx.eval(embedded)
    del embed_weight
    n = int(windows.shape[0])
    if model_type == "qwen4_exp":
        embedded, layer_masks, pos = oq._streamed_qwen4_exp_state(config, windows, embedded)
        mx.eval(embedded)
        states = [oq._streamed_qwen4_exp_slot_state(windows, pos, i, i + 1) for i in range(n)]
        masks = layer_masks
    else:
        _, lm, pos = oq._prepare_layer_inputs(oq._StreamingNoModel(), [None], windows[:1], embedded[:1],
                                              noncausal=oq._noncausal_attention(config))
        states, masks = [pos] * n, None
    working = [embedded[i:i + 1] for i in range(n)]
    del embedded

    route_idx, route_p, moe_layers = [], [], []
    for layer_idx, block, _ in oq._iter_streamed_layer_blocks(source, config,
                                                               trust_remote_code=trust_remote_code):
        router = _find_router(block) if router_k else None
        tap = _RouterTap(router_k) if router is not None else None
        if tap is not None:
            tap.attach(router)
            tap.record_as = "teacher"
        try:
            for s in range(n):
                if tap is not None:
                    tap.slot = s
                m = masks[layer_idx] if masks is not None else lm[0]
                out, _ = oq._forward_layer_result(block, working[s], m, states[s], layer_idx=layer_idx)
                mx.eval(out)
                working[s] = out
        finally:
            if tap is not None:
                tap.detach()
        if tap is not None and tap.teacher:
            moe_layers.append(layer_idx)
            route_idx.append(np.stack([np.array(tap.teacher[s][0]).astype(np.int16) for s in range(n)]))
            route_p.append(np.stack([np.array(tap.teacher[s][1]).astype(np.float16) for s in range(n)]))
        del block
        mx.clear_cache()
        if progress:
            progress(layer_idx)

    head = _streamed_head(source, config)
    ids_all, top_all, rest_all = [], [], []
    for s in range(n):
        h = working[s][0]
        ids_w, top_w, rest_w = [], [], []
        for t0 in range(0, h.shape[0], token_chunk):
            i_, t_, r_ = _topk_logprobs(head(h[None, t0:t0 + token_chunk])[0], top_k)
            mx.eval(i_, t_, r_)
            ids_w.append(np.array(i_).astype(np.int32))
            top_w.append(np.array(t_).astype(np.float16))
            rest_w.append(np.array(r_).astype(np.float32))
        ids_all.append(np.concatenate(ids_w))
        top_all.append(np.concatenate(top_w))
        rest_all.append(np.concatenate(rest_w))
        mx.clear_cache()
    meta = {"version": REFERENCE_VERSION, "source": str(source.resolve()), "top_k": top_k,
            "router_k": router_k, "windows": n, "seq_len": int(windows.shape[1]),
            "slices": slices or ["all"] * n, "moe_layers": moe_layers}
    arrays = {"windows": np.array(windows).astype(np.int32), "top_ids": np.stack(ids_all),
              "top_logp": np.stack(top_all), "rest_logp": np.stack(rest_all)}
    if moe_layers:
        arrays["route_idx"] = np.stack(route_idx)       # [layers, N, L, k]
        arrays["route_p"] = np.stack(route_p)
    tmp = Path(out_path).with_suffix(".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(out_path)
    return meta


def _model_layers(model):
    for path in ("language_model.model.layers", "model.layers", "layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError("cannot find the decoder layers of the loaded model")


def _forward_window_logits(model, ids: mx.array) -> mx.array:
    arr = ids[None]
    for call in (lambda: model.language_model(arr), lambda: model(arr)):
        try:
            out = call()
            break
        except (TypeError, AttributeError):
            continue
    else:
        raise RuntimeError("could not run a forward pass on the loaded model")
    logits = getattr(out, "logits", out)
    return logits[0]


def score_against_reference(model, reference_path: str, *, token_chunk: int = 512) -> dict:
    """KL, top-1 agreement and routing flips of a loaded model vs the reference."""
    data = np.load(reference_path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    windows = data["windows"]
    n = windows.shape[0]
    layers = _model_layers(model)
    moe_layers = meta["moe_layers"]
    taps = {}
    for li in moe_layers:
        r = _find_router(layers[li])
        if r is not None:
            tap = _RouterTap(meta["router_k"])
            tap.attach(r)
            tap.record_as = "student"
            taps[li] = tap
    per_slice: dict[str, dict] = {}
    layer_shift = {li: [0.0, 0.0, 0] for li in taps}          # mass, flips, tokens
    try:
        for s in range(n):
            for tap in taps.values():
                tap.slot = s
            logits = _forward_window_logits(model, mx.array(windows[s]))
            mx.eval(logits)
            ids = mx.array(data["top_ids"][s])
            tlogp = mx.array(data["top_logp"][s].astype(np.float32))
            trest = mx.array(data["rest_logp"][s])
            kl_sum = agree = tokens = 0.0
            for t0 in range(0, logits.shape[0], token_chunk):
                lg = logits[t0:t0 + token_chunk].astype(mx.float32)
                logq = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                i_, tp, tr = ids[t0:t0 + token_chunk], tlogp[t0:t0 + token_chunk], trest[t0:t0 + token_chunk]
                q_top = mx.take_along_axis(logq, i_, axis=-1)
                q_rest = mx.log(mx.maximum(1.0 - mx.exp(q_top).sum(axis=-1), 1e-12))
                p = mx.exp(tp)
                kl = (p * (tp - q_top)).sum(axis=-1) + mx.exp(tr) * (tr - q_rest)
                kl_sum += float(mx.sum(mx.maximum(kl, 0.0)).item())
                agree += float(mx.sum(mx.argmax(logq, axis=-1) == i_[:, 0]).item())
                tokens += lg.shape[0]
            sl = meta["slices"][s]
            acc = per_slice.setdefault(sl, {"kl_sum": 0.0, "agree": 0.0, "tokens": 0.0})
            acc["kl_sum"] += kl_sum
            acc["agree"] += agree
            acc["tokens"] += tokens
            for j, li in enumerate(moe_layers):
                tap = taps.get(li)
                if tap is None or s not in tap.student:
                    continue
                t_idx = mx.array(data["route_idx"][j, s].astype(np.int32))
                t_p = mx.array(data["route_p"][j, s].astype(np.float32))
                s_idx = tap.student[s]
                kept = mx.any(t_idx[:, :, None] == s_idx[:, None, :], axis=-1)
                layer_shift[li][0] += float(mx.sum(mx.sum(t_p * (1.0 - kept.astype(mx.float32)), axis=-1)).item())
                layer_shift[li][1] += float(mx.sum(mx.any(~kept, axis=-1).astype(mx.float32)).item())
                layer_shift[li][2] += int(t_idx.shape[0])
            for tap in taps.values():
                tap.reset_student()
            mx.clear_cache()
    finally:
        for tap in taps.values():
            tap.detach()
    slices = {k: {"kl": v["kl_sum"] / max(v["tokens"], 1), "top1": v["agree"] / max(v["tokens"], 1),
                  "tokens": int(v["tokens"])} for k, v in sorted(per_slice.items())}
    total_tokens = sum(v["tokens"] for v in per_slice.values())
    overall = {"kl": sum(v["kl_sum"] for v in per_slice.values()) / max(total_tokens, 1),
               "top1": sum(v["agree"] for v in per_slice.values()) / max(total_tokens, 1),
               "tokens": int(total_tokens)}
    routing = {str(li): {"mass_shift": m / max(t, 1), "flip_rate": f / max(t, 1)}
               for li, (m, f, t) in layer_shift.items() if t}
    if routing:
        overall["routing_mass_shift"] = float(np.mean([v["mass_shift"] for v in routing.values()]))
        overall["routing_flip_rate"] = float(np.mean([v["flip_rate"] for v in routing.values()]))
    return {"reference": meta, "overall": overall, "slices": slices, "routing_by_layer": routing}
