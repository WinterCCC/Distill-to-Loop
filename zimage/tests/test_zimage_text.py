"""The prompt encoder around ``ZImagePipeline._encode_prompt`` and the chunking of ``encode_prompts``, on a stand-in pipeline (no text
encoder weights)."""

from __future__ import annotations

import pytest
import torch

from distill_to_loop.data.text import ATTN_MASK, POOLED, TEXT_EMB
from dooop_zimage.inference import ENCODE_CHUNK, encode_prompts
from dooop_zimage.text import ZImagePromptEncoder
from zimage_tiny import TEXT_PIPE_WIDTH as WIDTH, FakeTextPipe as _Pipe


def test_training_cache_form():
    pipe = _Pipe()
    encoder = ZImagePromptEncoder(pipe, stored_rows=6)
    captions = ["a b c", "a b c d e f"]
    out = encoder(captions)
    assert captions == ["a b c", "a b c d e f"]  # the caller's list is untouched
    assert out[TEXT_EMB].shape == (2, 6, WIDTH) and out[TEXT_EMB].dtype == torch.bfloat16
    assert out[POOLED].tolist() == [[1, 1, 1, 0, 0, 0], [1] * 6] and out[POOLED].dtype == torch.bfloat16
    assert torch.equal(out[ATTN_MASK], out[POOLED].bool())
    assert torch.count_nonzero(out[TEXT_EMB][0, 3:]) == 0
    with pytest.raises(ValueError, match="more than 6 tokens"):
        encoder(["a b c d e f g"])


def test_a_prompt_the_pipeline_would_truncate_is_refused():
    encoder = ZImagePromptEncoder(_Pipe(), stored_rows=4, max_sequence_length=4)
    assert [r.shape[0] for r in encoder.rows(["a b c d"])] == [4]  # exactly at the limit: kept
    with pytest.raises(ValueError, match="more than 4 tokens, which the pipeline truncates"):
        encoder.rows(["a b c d e"])


def test_encode_prompts_chunks_with_the_empty_prompt_last():
    pipe = _Pipe()
    texts = [f"prompt {i}" for i in range(ENCODE_CHUNK + 1)]
    out = encode_prompts(texts, "cpu", encoder=ZImagePromptEncoder(pipe))
    assert pipe.calls == [texts[:ENCODE_CHUNK], [texts[-1], ""]]
    assert len(out["prompt_embeds"]) == len(texts) and out["prompt_embeds"][3].shape == (2, WIDTH)
    assert out["negative_prompt_embeds"].shape == (1, WIDTH) and out["negative_prompt_embeds"][0, 0] == 0.0
