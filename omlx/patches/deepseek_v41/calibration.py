# SPDX-License-Identifier: MIT
"""Collect oQe statistics through V4.1's complete cache and Engram forward."""

import copy
import json
import logging
from pathlib import Path

import mlx.core as mx

from .cache import DeepseekV41Cache
from .config import ModelConfig

logger = logging.getLogger(__name__)


def collect_imatrix(
    model,
    tokenizer,
    *,
    num_samples=32,
    seq_length=512,
    progress=None,
    initial=None,
    calib_dataset="oqe_code_multilingual",
    adaptive=False,
):
    from ...oq import (
        _OQE_MAX_ADAPTIVE_SAMPLES,
        _OQE_MAX_SAMPLE_MULTIPLIER,
        OQImatrixCollector,
        _imatrix_expert_coverage_stats,
        _imatrix_expert_coverage_sufficient,
        _load_calibration_data,
    )

    maximum = num_samples
    if adaptive:
        maximum = max(
            num_samples,
            min(num_samples * _OQE_MAX_SAMPLE_MULTIPLIER, _OQE_MAX_ADAPTIVE_SAMPLES),
        )
    samples = _load_calibration_data(tokenizer, calib_dataset, maximum, seq_length)
    if samples is None or samples.shape[0] == 0:
        raise ValueError("V4.1 oQe requires nonempty calibration data")
    collector = OQImatrixCollector()
    start = 0
    if initial is not None:
        expected = {
            "model_type": "deepseek_v41",
            "dataset": calib_dataset,
            "seq_length": seq_length,
        }
        if any(initial.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("V4.1 calibration resume metadata does not match")
        start = int(initial.metadata.get("processed_samples", -1))
        if not 0 <= start < len(samples):
            raise ValueError("V4.1 calibration resume has no remaining samples")
        collector.entries = copy.deepcopy(initial.entries)
    core = model.language_model
    installed = collector.install(model)
    processed = start
    rounds = []
    try:
        for index, tokens in enumerate(samples[start:], start=start):
            cache = core.make_cache()
            logits = core(tokens[None], cache=cache)
            mx.eval(logits[:, -1], [entry.state for entry in cache])
            if progress is not None:
                progress(index + 1, len(samples), collector.entries)
            del logits, cache
            processed = index + 1
            if adaptive and processed >= num_samples and processed % num_samples == 0:
                coverage = _imatrix_expert_coverage_stats(collector.entries)
                sufficient = _imatrix_expert_coverage_sufficient(
                    coverage, require_expert_counts=collector.switch_capture_modules > 0
                )
                rounds.append(dict(processed_samples=processed, coverage=coverage))
                if sufficient:
                    break
        if not collector.entries:
            raise ValueError("V4.1 oQe captured no projection inputs")
        metadata = {
            "model_type": "deepseek_v41",
            "dataset": calib_dataset,
            "processed_samples": processed,
            "requested_samples": num_samples,
            "adaptive_max_samples": len(samples),
            "rounds": rounds,
            "seq_length": seq_length,
            "capture_modules": installed,
            "captured_modules": len(collector.entries),
            "expert_coverage": _imatrix_expert_coverage_stats(collector.entries),
            "forward": "full_model_with_request_local_cache_and_engram",
        }
        return collector.entries, metadata
    finally:
        collector.restore(model)


def collect_checkpoint_imatrix(
    model_path,
    *,
    calib_dataset,
    num_samples,
    seq_length,
    progress_callback=None,
    progress_start=13.0,
    progress_end=18.0,
):
    """Adapter for the official oQ collector, including cancellation cleanup."""
    from ...oq import _emit_progress, _imatrix_expert_coverage_sufficient
    from .loading import load

    model, processor = load(model_path, engram_ssd_offload=True)
    try:

        def progress(done, total, entries):
            _emit_progress(
                progress_callback,
                "imatrix",
                progress_start + (progress_end - progress_start) * done / total,
                f"V4.1 calibration {done}/{total}",
                {"entry_count": len(entries), "processed_samples": done},
            )

        entries, metadata = collect_imatrix(
            model,
            processor.tokenizer,
            calib_dataset=calib_dataset,
            num_samples=num_samples,
            seq_length=seq_length,
            progress=progress,
            adaptive=True,
        )
        coverage = metadata["expert_coverage"]
        sufficient = _imatrix_expert_coverage_sufficient(
            coverage, require_expert_counts=True
        )
        metadata.update(
            coverage=coverage,
            requires_expert_counts=True,
            coverage_sufficient=sufficient,
            collection_sufficient=sufficient,
            uncalibrated_policy="preserve_source_precision",
        )
        return entries, metadata
    finally:
        model.close()
        del model, processor
        mx.synchronize()
        mx.clear_cache()


def _v41_source_mapping(source):
    return json.loads(
        (Path(source) / "model.safetensors.index.json").read_text()
    )["weight_map"]


def _load_v41_embed_weight(source, config, mapping):
    """Read and dequantize the token embedding without touching a decoder layer."""
    from .convert import iter_source_weights

    weight = None
    for values, _ in iter_source_weights(
        source,
        config,
        mapping,
        key_filter=lambda key: key.split(".", 1)[0] == "embed",
    ):
        weight = values.get("language_model.embed.weight")
    if weight is None:
        raise ValueError("V4.1 checkpoint has no token embedding")
    mx.eval(weight)
    return weight


def _close_v41_block(block):
    engram = getattr(block, "engram", None)
    embed = getattr(engram, "embed", None)
    close = getattr(embed, "close", None)
    if close is not None:
        close()


def _load_v41_layer_block(source, config, mapping, tables, layer_idx):
    """Build one decoder block and fill it from that layer's source shards.

    The block owns the layer's packed routed experts (about 7-10 GB on the
    released checkpoint) but no other layer. Engram tables stay on disk and
    are reached through ``DiskEngramEmbedding``.
    """
    from .convert import iter_source_weights
    from .language import Block
    from .loading import set_module
    from .quantization import QuantizedProjection
    from .storage import DiskEngramEmbedding

    model_config = ModelConfig.from_dict(config)
    block = Block(model_config, layer_idx)
    prefix = f"language_model.layers.{layer_idx}."
    table = tables.get(prefix + "engram.embed")
    if layer_idx in model_config.engram_layer_ids and table is None:
        raise ValueError(f"V4.1 Engram layer {layer_idx} has no source table")
    if table is not None:
        if not hasattr(block, "engram"):
            raise ValueError(f"V4.1 layer {layer_idx} has no Engram module")
        block.engram.embed = DiskEngramEmbedding(
            Path(source) / table["weight_file"],
            table["weight_key"],
            table["scale_key"],
            (
                Path(source) / table["scale_file"]
                if table.get("scale_file")
                else None
            ),
            bias_key=table.get("bias_key"),
            bits=table.get("bits"),
            group_size=table.get("group_size") or 32,
        )
    dense = {}
    try:
        for values, specs in iter_source_weights(
            source,
            config,
            mapping,
            key_filter=lambda key: key.startswith(f"layers.{layer_idx}."),
        ):
            for name, spec in specs.items():
                if not name.startswith(prefix):
                    continue
                fields = {
                    field: values[f"{name}.{field}"]
                    for field in ("weight", "scales", "biases")
                    if f"{name}.{field}" in values
                }
                if "weight" not in fields or "scales" not in fields:
                    raise ValueError(f"Packed V4.1 projection is incomplete: {name}")
                set_module(
                    block,
                    name[len(prefix) :],
                    QuantizedProjection(**fields, **spec),
                )
            for key, value in values.items():
                if not key.startswith(prefix):
                    continue
                module = key.rsplit(".", 1)[0]
                if module in specs:
                    continue
                dense[key[len(prefix) :]] = value
        block.load_weights(list(dense.items()), strict=False)
        mx.eval(block.parameters())
        block.eval()
        return block
    except Exception:
        _close_v41_block(block)
        raise


def _v41_hasher(model_config, tokenizer):
    if not model_config.engram_layer_ids:
        return None
    from .engram import NgramHash, build_compressed_token_map

    token_map, _ = build_compressed_token_map(tokenizer)
    return NgramHash(model_config, token_map)


class _V41StreamState:
    """Per-sample V4.1 request state for a layer-outer calibration sweep.

    ``h``/``pre``/``shared`` mirror the row loop in LanguageModel._forward;
    ``caches`` holds one request-local DeepseekV41Cache per layer. A layer's
    cache is released as soon as that layer has run: only the compressed KV
    copies kept in ``shared`` are needed by later layers.
    """

    def __init__(self, model_config, tokens, embed_weight, hasher):
        import numpy as np

        self.config = model_config
        self.tokens = np.asarray(tokens, dtype=np.int64)
        if self.tokens.ndim != 2 or self.tokens.shape[0] == 0:
            raise ValueError("V4.1 streaming calibration requires token rows")
        self.seq_length = int(self.tokens.shape[1])
        self.n_layers = int(model_config.n_layers)
        self.engram_index = {
            layer: index
            for index, layer in enumerate(model_config.engram_layer_ids)
        }
        hashes = None if hasher is None else hasher(self.tokens)
        self.hashes = None if hashes is None else hashes[0]
        h = embed_weight[mx.array(self.tokens, dtype=mx.int32)]
        h = mx.repeat(h[..., None, :], model_config.hc_mult, -2)
        pre = mx.broadcast_to(
            (mx.arange(model_config.hc_mult) == 0).astype(mx.float32),
            h.shape[:-1],
        )
        mx.eval(h, pre)
        self.h = [h[i : i + 1] for i in range(self.tokens.shape[0])]
        self.pre = [pre[i : i + 1] for i in range(self.tokens.shape[0])]
        self.caches = [
            [
                DeepseekV41Cache(
                    model_config.compress_ratios[layer]
                    if layer in model_config.kv_source_layers
                    else 0
                )
                for layer in range(self.n_layers)
            ]
            for _ in range(self.tokens.shape[0])
        ]
        self.shared = [{} for _ in range(self.tokens.shape[0])]
        self.scores = {}

    def _empty_cache(self, layer_idx):
        ratio = (
            self.config.compress_ratios[layer_idx]
            if layer_idx in self.config.kv_source_layers
            else 0
        )
        return DeepseekV41Cache(ratio)

    def advance_layer(self, block, layer_idx):
        """Run one decoder layer for every sample, handing h/pre to the next."""
        engram_ix = self.engram_index.get(layer_idx)
        for sample in range(len(self.h)):
            h = self.h[sample]
            if engram_ix is not None:
                if self.hashes is None:
                    raise ValueError("V4.1 Engram layer without an n-gram hasher")
                h = block.engram(
                    h, self.hashes[sample : sample + 1, :, engram_ix], None
                )
            cache = self.caches[sample][layer_idx]
            if cache is None:
                raise RuntimeError(f"V4.1 layer {layer_idx} cache was already released")
            out_h, out_pre = block(
                h,
                self.pre[sample],
                cache,
                self.shared[sample],
                0,
                None,
            )
            state = [value for value in cache.cache if value is not None]
            mx.eval(out_h, out_pre, state)
            cache[0] = mx.array([self.seq_length], dtype=mx.int32)
            self.h[sample] = out_h
            self.pre[sample] = out_pre
            self.caches[sample][layer_idx] = None

    def measure_layer(self, block, layer_idx, bits):
        """Measure this layer's qdq distortion on the sensitivity boundary."""
        from .sensitivity import _quantized_block

        engram_ix = self.engram_index.get(layer_idx)
        baselines = []
        inputs = []
        for sample in range(len(self.h)):
            h = self.h[sample]
            if engram_ix is not None:
                if self.hashes is None:
                    raise ValueError("V4.1 Engram layer without an n-gram hasher")
                h = block.engram(
                    h, self.hashes[sample : sample + 1, :, engram_ix], None
                )
            cache = self.caches[sample][layer_idx]
            if cache is None:
                raise RuntimeError(f"V4.1 layer {layer_idx} cache was already released")
            shared_input = dict(self.shared[sample])
            out = block(
                h,
                self.pre[sample],
                cache,
                self.shared[sample],
                0,
                None,
            )
            state = [value for value in cache.cache if value is not None]
            mx.eval(out, state)
            cache[0] = mx.array([self.seq_length], dtype=mx.int32)
            self.caches[sample][layer_idx] = None
            baselines.append(out)
            inputs.append((h, self.pre[sample], shared_input))

        with _quantized_block(block, bits) as changed:
            if not changed:
                raise ValueError(
                    f"No quantizable projections in V4.1 layer {layer_idx}"
                )
            error = power = 0.0
            for (h, pre, shared_input), reference in zip(inputs, baselines):
                actual = block(
                    h,
                    pre,
                    self._empty_cache(layer_idx),
                    dict(shared_input),
                    0,
                    None,
                )
                mx.eval(actual)
                for original, converted in zip(reference, actual):
                    original = original.astype(mx.float32)
                    converted = converted.astype(mx.float32)
                    error += mx.sum(mx.square(original - converted)).item()
                    power += mx.sum(mx.square(original)).item()
        self.scores[layer_idx] = error / max(power, 1e-10)
        for sample, (out_h, out_pre) in enumerate(baselines):
            self.h[sample] = out_h
            self.pre[sample] = out_pre


def _v41_sensitivity_tokens(tokenizer, dataset, num_samples, seq_length, seed):
    from ...oq import _load_calibration_data

    mx.random.seed(int(seed))
    tokens = _load_calibration_data(
        tokenizer, dataset=dataset, num_samples=num_samples, seq_length=seq_length
    )
    if tokens is None or len(tokens) == 0:
        return None
    return tokens


def collect_checkpoint_imatrix_streaming(
    model_path,
    tokenizer,
    config,
    *,
    calib_dataset,
    num_samples,
    seq_length,
    progress_callback=None,
    progress_start=13.0,
    progress_end=18.0,
    trust_remote_code=False,
    calib_data=None,
    calib_seed=0,
    measure_sensitivity=False,
    sensitivity_oq_level=None,
    sensitivity_calib_dataset="code_multilingual",
    sensitivity_num_samples=32,
    sensitivity_seq_length=256,
    sensitivity_calib_seed=0,
    require_mtp_entries=True,
):
    """Collect the V4.1 imatrix with one decoder block resident at a time.

    The layer loop reproduces the cache, Engram, shared-KV and hyper-connection
    state of ``LanguageModel._forward``, but sources each ``Block`` directly
    from the original checkpoint. Adaptive rounds re-sweep only after expert
    coverage is insufficient; sensitivity is fused into the first round so one
    weight read serves both measurements.
    """
    from ...oq import (
        _OQE_MAX_ADAPTIVE_SAMPLES,
        _OQE_MAX_SAMPLE_MULTIPLIER,
        OQImatrixCollector,
        _base_bits_for_level,
        _emit_progress,
        _format_size,
        _imatrix_expert_coverage_stats,
        _imatrix_expert_coverage_sufficient,
        _imatrix_requires_expert_counts,
        _load_calibration_data,
        _log_streamed_sensitivity,
    )
    from .convert import source_engram_tables

    if measure_sensitivity and sensitivity_oq_level is None:
        raise ValueError("measure_sensitivity requires sensitivity_oq_level")
    source = Path(model_path)
    if tokenizer is None:
        from mlx_lm.tokenizer_utils import load as load_tokenizer

        tokenizer = load_tokenizer(source)
    mapping = _v41_source_mapping(source)
    model_config = ModelConfig.from_dict(config)
    adaptive_max = max(
        int(num_samples),
        min(
            int(num_samples) * _OQE_MAX_SAMPLE_MULTIPLIER,
            _OQE_MAX_ADAPTIVE_SAMPLES,
        ),
    )
    if calib_data is None:
        mx.random.seed(int(calib_seed))
        calib_data = _load_calibration_data(
            tokenizer,
            dataset=calib_dataset,
            num_samples=adaptive_max,
            seq_length=seq_length,
        )
    if calib_data is None or len(calib_data) == 0:
        return {}, {
            "dataset": calib_dataset,
            "processed_samples": 0,
            "failure_stage": "calibration-data",
            "failure_reason": "empty V4.1 calibration draw",
        }
    import numpy as np

    calib_data = np.asarray(calib_data, dtype=np.int64)
    available_samples = int(calib_data.shape[0])
    max_samples = min(available_samples, adaptive_max)
    embed_weight = _load_v41_embed_weight(source, config, mapping)
    hasher = _v41_hasher(model_config, tokenizer)
    tables = source_engram_tables(mapping, config)
    sensitivity_bits = (
        _base_bits_for_level(sensitivity_oq_level)
        if measure_sensitivity
        else None
    )
    sensitivity_tokens = None
    if measure_sensitivity:
        sensitivity_tokens = _v41_sensitivity_tokens(
            tokenizer,
            sensitivity_calib_dataset,
            sensitivity_num_samples,
            sensitivity_seq_length,
            sensitivity_calib_seed,
        )
        if sensitivity_tokens is None:
            logger.warning(
                "V4.1 streaming calibration: sensitivity draw failed; "
                "the sweep will return an empty sensitivity map"
            )
    collector = OQImatrixCollector(
        skip=config.get("_oq_imatrix_capture_skip", frozenset())
    )
    installed = 0
    switch_capture_modules = 0
    capture_module_classes = {}
    require_expert_counts = False
    coverage = _imatrix_expert_coverage_stats(collector.entries)
    coverage_sufficient = False
    collection_sufficient = False
    processed_samples = 0
    round_index = 0
    rounds = []
    scores = {}
    total_layers = model_config.n_layers

    logger.info(
        "V4.1 imatrix streaming: adaptive max=%d, step=%d, layers=%d",
        max_samples,
        int(num_samples),
        total_layers,
    )

    while processed_samples < max_samples:
        round_start = processed_samples
        round_stop = min(round_start + int(num_samples), max_samples)
        state = _V41StreamState(
            model_config,
            calib_data[round_start:round_stop],
            embed_weight,
            hasher,
        )
        sens_state = None
        if sensitivity_tokens is not None and round_index == 0:
            sens_state = _V41StreamState(
                model_config, sensitivity_tokens, embed_weight, hasher
            )
        for layer_idx in range(total_layers):
            block = _load_v41_layer_block(
                source, config, mapping, tables, layer_idx
            )
            try:
                layer_installed = collector.install(
                    block,
                    name_prefix=f"language_model.layers.{layer_idx}.",
                )
                if round_index == 0:
                    installed += layer_installed
                    if layer_idx == 0 and layer_installed == 0:
                        raise RuntimeError(
                            "V4.1 streamed layer 0 installed 0 capture modules; "
                            "the source layout does not match the capture predicate"
                        )
                state.advance_layer(block, layer_idx)
                collector.restore(block)
                if sens_state is not None:
                    sens_state.measure_layer(
                        block, layer_idx, sensitivity_bits
                    )
            finally:
                collector.restore(block)
                _close_v41_block(block)
                del block
                mx.synchronize()
                mx.clear_cache()
            active = mx.get_active_memory()
            logger.info(
                "V4.1 imatrix streaming: round %d layer %d/%d done "
                "(active %.1f GB, peak %.1f GB)",
                round_index + 1,
                layer_idx + 1,
                total_layers,
                active / 1e9,
                mx.get_peak_memory() / 1e9,
            )
            if active > 50 * 1024**3:
                raise RuntimeError(
                    f"V4.1 streamed layer {layer_idx} left "
                    f"{active / 1e9:.1f} GB active, past the 50 GB "
                    "streaming budget; aborting before the leak compounds"
                )
            layer_frac = (layer_idx + 1) / max(total_layers, 1)
            frac = (
                round_start + (round_stop - round_start) * layer_frac
            ) / max(max_samples, 1)
            _emit_progress(
                progress_callback,
                "imatrix",
                progress_start
                + (progress_end - progress_start) * min(max(frac, 0.0), 1.0),
                f"V4.1 imatrix streaming round {round_index + 1}: "
                f"layer {layer_idx + 1}/{total_layers}",
                {
                    "round": round_index + 1,
                    "layer": layer_idx,
                    "processed_samples": round_start,
                    "round_samples": round_stop - round_start,
                    "max_samples": max_samples,
                    "active_memory_bytes": int(active),
                },
            )
        if round_index == 0:
            capture_module_classes = dict(collector.capture_module_classes)
            switch_capture_modules = int(collector.switch_capture_modules)
            require_expert_counts = _imatrix_requires_expert_counts(
                config, switch_capture_modules
            )
            if sens_state is not None:
                scores = dict(sens_state.scores)
                _log_streamed_sensitivity(sensitivity_oq_level, scores)
        del state, sens_state
        mx.clear_cache()
        processed_samples = round_stop
        round_index += 1
        coverage = _imatrix_expert_coverage_stats(collector.entries)
        coverage_sufficient = _imatrix_expert_coverage_sufficient(
            coverage, require_expert_counts=require_expert_counts
        )
        collection_sufficient = (
            processed_samples >= int(num_samples) and coverage_sufficient
        )
        rounds.append(
            {
                "processed_samples": processed_samples,
                "coverage_sufficient": coverage_sufficient,
                "collection_sufficient": collection_sufficient,
                "coverage": coverage,
            }
        )
        logger.info(
            "V4.1 imatrix streaming: round %d done, %d/%d samples, "
            "zero experts=%d, sufficient=%s",
            round_index,
            processed_samples,
            max_samples,
            int(coverage.get("zero_count_experts", 0)),
            collection_sufficient,
        )
        if collection_sufficient:
            break

    if not collector.entries:
        raise ValueError("V4.1 streaming calibration captured no projection inputs")
    metadata = {
        "model_type": "deepseek_v41",
        "dataset": calib_dataset,
        "requested_samples": int(num_samples),
        "seq_length": int(seq_length),
        "adaptive": True,
        "adaptive_max_samples": max_samples,
        "available_samples": available_samples,
        "processed_samples": processed_samples,
        "rounds": rounds,
        "capture_modules": installed,
        "captured_modules": len(collector.entries),
        "capture_module_classes": dict(sorted(capture_module_classes.items())),
        "switch_capture_modules": switch_capture_modules,
        "requires_expert_counts": require_expert_counts,
        "coverage": coverage,
        "expert_coverage": coverage,
        "coverage_sufficient": coverage_sufficient,
        "collection_sufficient": collection_sufficient,
        "load_kind": "streaming",
        "forward": "layer_streaming_with_request_local_cache_and_engram",
        "uncalibrated_policy": "preserve_source_precision",
        "streaming_batch_plan": {
            "samples_per_round": int(num_samples),
            "resident_layers": 1,
        },
        "streaming_source_bytes": _format_size(
            sum(
                path.stat().st_size
                for path in {Path(source) / name for name in mapping.values()}
            )
        ),
    }
    if measure_sensitivity:
        metadata["sensitivity_map"] = scores
    return collector.entries, metadata


def measure_sensitivity_streaming(
    model_path,
    config,
    *,
    bits,
    calib_dataset="code_multilingual",
    num_samples=32,
    seq_length=256,
    calib_seed=0,
    tokenizer=None,
    progress_callback=None,
    progress_start=10.0,
    progress_end=20.0,
):
    """Measure V4.1 layer sensitivity without loading the whole checkpoint."""
    from ...oq import _emit_progress, _load_calibration_data
    from .convert import source_engram_tables

    source = Path(model_path)
    if tokenizer is None:
        from mlx_lm.tokenizer_utils import load as load_tokenizer

        tokenizer = load_tokenizer(source)
    mapping = _v41_source_mapping(source)
    model_config = ModelConfig.from_dict(config)
    mx.random.seed(int(calib_seed))
    tokens = _load_calibration_data(
        tokenizer, dataset=calib_dataset, num_samples=num_samples, seq_length=seq_length
    )
    if tokens is None or len(tokens) == 0:
        raise ValueError("V4.1 sensitivity requires nonempty calibration data")
    import numpy as np

    tokens = np.asarray(tokens, dtype=np.int64)
    embed_weight = _load_v41_embed_weight(source, config, mapping)
    hasher = _v41_hasher(model_config, tokenizer)
    tables = source_engram_tables(mapping, config)
    state = _V41StreamState(model_config, tokens, embed_weight, hasher)
    total = model_config.n_layers
    for layer_idx in range(total):
        block = _load_v41_layer_block(source, config, mapping, tables, layer_idx)
        try:
            state.measure_layer(block, layer_idx, bits)
        finally:
            _close_v41_block(block)
            del block
            mx.synchronize()
            mx.clear_cache()
        _emit_progress(
            progress_callback,
            "sensitivity",
            progress_start
            + (progress_end - progress_start) * (layer_idx + 1) / max(total, 1),
            f"V4.1 streamed sensitivity {layer_idx + 1}/{total}",
        )
        active = mx.get_active_memory()
        if active > 50 * 1024**3:
            raise RuntimeError(
                f"V4.1 streamed sensitivity layer {layer_idx} left "
                f"{active / 1e9:.1f} GB active, past the 50 GB streaming budget"
            )
    return state.scores
