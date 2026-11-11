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

import logging
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)

_VALID_BITS = (2, 3, 4, 5, 6, 8)
_CLIP_FACTORS = (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25)


# ---------------------------------------------------------------------------
# Hessians
# ---------------------------------------------------------------------------


@dataclass
class HessianAccumulator:
    """Running ``Σ x xᵀ`` and token count for one input space.

    ``batch`` > 0 keeps one accumulator per expert (``[E, in, in]``).
    """

    in_dim: int
    batch: int = 0

    def __post_init__(self):
        shape = (self.batch, self.in_dim, self.in_dim) if self.batch else (self.in_dim, self.in_dim)
        self.sum_xxT = mx.zeros(shape, dtype=mx.float32)
        self.count = mx.zeros((self.batch,) if self.batch else (), dtype=mx.float32)

    def add(self, x: mx.array) -> None:
        """Dense input: ``x[..., in_dim]``."""
        x = x.reshape(-1, self.in_dim).astype(mx.float32)
        self.sum_xxT = self.sum_xxT + x.T @ x
        self.count = self.count + x.shape[0]

    def add_routed(self, x: mx.array, expert_ids: mx.array) -> None:
        """Routed input: ``x[T, in_dim]`` with ``expert_ids[T, k]``.

        Each token contributes to every expert it was routed to. Done as one
        scatter-free matmul per expert chunk via a one-hot routing matrix.
        """
        x = x.reshape(-1, self.in_dim).astype(mx.float32)
        ids = expert_ids.reshape(x.shape[0], -1)
        onehot = mx.zeros((x.shape[0], self.batch), dtype=mx.float32)
        for k in range(ids.shape[1]):
            onehot = onehot + (ids[:, k : k + 1] == mx.arange(self.batch)[None, :]).astype(mx.float32)
        # Σ_t onehot[t,e] x_t x_tᵀ  ==  (x * onehot[:, e])ᵀ x, batched over e.
        weighted = onehot.T[:, :, None] * x[None, :, :]          # [E, T, in]
        self.sum_xxT = self.sum_xxT + weighted.swapaxes(-1, -2) @ x[None]
        self.count = self.count + onehot.sum(axis=0)

    def mean(self) -> mx.array:
        """Per-token average ``E[x xᵀ]`` (zero where no tokens were seen)."""
        c = mx.maximum(self.count, 1.0)
        if self.batch:
            return self.sum_xxT / c[:, None, None]
        return self.sum_xxT / c


def shrink_expert_hessians(
    sum_xxT: mx.array, counts: mx.array, prior_tokens: float = 256.0
) -> mx.array:
    """Blend each expert's Hessian toward the layer-wide average.

    ``H_e = (S_e + λ · H̄) / (n_e + λ)`` where ``S_e = Σ x xᵀ`` over the
    expert's tokens, ``H̄`` the per-token average over all routed tokens and
    ``λ = prior_tokens``. A well-covered expert (``n_e ≫ λ``) keeps its own
    statistics; an unseen expert gets the layer average instead of a zero
    matrix.
    """
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
    sum_xxT: mx.array,
    counts: mx.array,
    bits: int,
    group_size: int = 64,
    *,
    prior_tokens: float = 256.0,
    expert_chunk: int = 32,
    **kwargs,
):
    """GPTQ for stacked experts ``w[E, out, in]`` with per-expert statistics.

    Hessians are shrunk toward the layer average (:func:`shrink_expert_hessians`)
    and processed ``expert_chunk`` experts at a time to bound memory
    (``chunk · in² · 4`` bytes for the Cholesky factors).
    """
    h = shrink_expert_hessians(sum_xxT, counts, prior_tokens)
    parts = []
    for e0 in range(0, w.shape[0], expert_chunk):
        e1 = min(e0 + expert_chunk, w.shape[0])
        parts.append(gptq_quantize(w[e0:e1], h[e0:e1], bits, group_size, **kwargs))
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
