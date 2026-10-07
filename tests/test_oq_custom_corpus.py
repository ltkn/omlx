# SPDX-License-Identifier: Apache-2.0
"""Tests for custom-corpus calibration and the qwen4_exp N-gram bit override."""

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.oq import (
    _get_predicate_bits,
    _load_calibration_data,
    _load_file_calibration,
    _pack_texts_to_samples,
    _source_imatrix_signature,
    _structural_quant_overrides,
    quantize_oq_streaming,
)

_NGRAM_SHARD = (
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
)
_NGRAM_BASE = _NGRAM_SHARD.removesuffix(".weight")


def _qwen4_config(**extra):
    config = {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": "qwen4_exp_text"},
    }
    config.update(extra)
    return config


class TestNgramBitsOverride:
    def test_default_keeps_base_bits_group32(self):
        bits, gs, mode = _get_predicate_bits(_NGRAM_SHARD, _qwen4_config(), 4, 64)

        assert (bits, gs, mode) == (4, 32, "affine")

    def test_override_q8_at_oq4(self):
        bits, gs, mode = _get_predicate_bits(
            _NGRAM_SHARD, _qwen4_config(_oq_ngram_bits=8), 4, 64
        )

        assert (bits, gs, mode) == (8, 32, "affine")

    def test_override_only_applies_to_ngram_table(self):
        config = _qwen4_config(_oq_ngram_bits=8)

        bits, gs, mode = _get_predicate_bits(
            "model.language_model.layers.16.self_attn.q_proj.weight",
            config,
            4,
            64,
        )

        assert (bits, gs, mode) == (4, 64, "affine")

    def test_structural_overrides_price_override_bits(self):
        named_shapes = {_NGRAM_BASE: (2_800, 160)}

        default = _structural_quant_overrides(named_shapes, _qwen4_config(), 4)
        boosted = _structural_quant_overrides(
            named_shapes, _qwen4_config(_oq_ngram_bits=8), 4
        )

        assert default[_NGRAM_BASE] == {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
        }
        assert boosted[_NGRAM_BASE] == {
            "bits": 8,
            "group_size": 32,
            "mode": "affine",
        }

    def test_quantize_rejects_invalid_ngram_bits(self):
        with pytest.raises(ValueError, match="ngram_bits"):
            quantize_oq_streaming(
                "/nonexistent-model", "/nonexistent-output", 4, ngram_bits=7
            )

    def test_quantize_rejects_ngram_bits_with_preserve_ngram_table(self):
        with pytest.raises(ValueError, match="preserve_ngram_table"):
            quantize_oq_streaming(
                "/nonexistent-model",
                "/nonexistent-output",
                4,
                ngram_bits=8,
                preserve_ngram_table=True,
            )


class TestCustomCorpusCacheSignature:
    def test_file_content_hash_invalidates_stale_cache(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("first version of the corpus", encoding="utf-8")
        config = {"model_type": "qwen4_exp"}
        kwargs = {"num_samples": 128, "seq_length": 512}

        before = _source_imatrix_signature(
            model_dir, config, calib_dataset=str(corpus), **kwargs
        )
        assert before["calib_data_hash"] != ""

        corpus.write_text("second version of the corpus", encoding="utf-8")
        after = _source_imatrix_signature(
            model_dir, config, calib_dataset=str(corpus), **kwargs
        )

        assert before["calib_data_hash"] != after["calib_data_hash"]

    def test_directory_hashes_member_files(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "01-a.txt").write_text("alpha", encoding="utf-8")
        (corpus_dir / "02-b.txt").write_text("beta", encoding="utf-8")
        config = {"model_type": "qwen4_exp"}
        kwargs = {"num_samples": 128, "seq_length": 512}

        sig = _source_imatrix_signature(
            model_dir, config, calib_dataset=str(corpus_dir), **kwargs
        )

        assert sig["calib_data_hash"] != ""


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCustomCorpusLoading:
    class _ByteTokenizer:
        def encode(self, text):
            return list(text.encode("utf-8"))

    def test_file_load_packs_samples(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("abcdefgh" * 64, encoding="utf-8")

        tokens = _load_file_calibration(
            self._ByteTokenizer(), corpus, num_samples=2, seq_length=64
        )

        assert tokens.shape == (2, 64)

    def test_directory_concatenates_txt_files(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "01-a.txt").write_text("abcdefgh" * 32, encoding="utf-8")
        (corpus_dir / "02-b.txt").write_text("ijklmnop" * 32, encoding="utf-8")

        tokens = _load_calibration_data(
            self._ByteTokenizer(),
            dataset=str(corpus_dir),
            num_samples=2,
            seq_length=64,
        )

        assert tokens.shape == (2, 64)

    def test_missing_tokens_raise(self, tmp_path):
        corpus = tmp_path / "tiny.txt"
        corpus.write_text("hi", encoding="utf-8")

        with pytest.raises(ValueError, match="Not enough tokens"):
            _load_file_calibration(
                self._ByteTokenizer(), corpus, num_samples=2, seq_length=64
            )

    def test_pack_helper_matches_builtin_path(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("abcdefgh" * 64, encoding="utf-8")
        tokenizer = self._ByteTokenizer()
        mx.random.seed(0)
        via_file = _load_file_calibration(
            tokenizer, corpus, num_samples=2, seq_length=64
        )
        mx.random.seed(0)
        via_helper = _pack_texts_to_samples(
            tokenizer, ["abcdefgh" * 64], num_samples=2, seq_length=64
        )

        assert mx.array_equal(via_file, via_helper).item()
