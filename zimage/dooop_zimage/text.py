"""Z-Image prompt encodings: ``ZImagePipeline._encode_prompt`` itself (the Qwen3 text encoder and its chat template).

The pipeline wraps a prompt in the Qwen3 chat template (``add_generation_prompt=True``, ``enable_thinking=True``), tokenizes it to at most 512
tokens and keeps the ``hidden_states[-2]`` rows of the real tokens, one ``[len_i, 2560]`` tensor per prompt; the transformer takes exactly that
list. :class:`ZImagePromptEncoder` calls that very method. For the training cache it stores each caption in :data:`STORED_TEXT_ROWS` rows (the
rows after its length zero) together with the validity mask (1 for a real token) in the ``pooled`` slot, which the student and the teacher
adapter read (:func:`dooop_zimage.model.caption_rows`); a caption of more rows raises. The unconditional embedding is the same encoding of the
empty prompt (the pipeline's default negative prompt).

A prompt of more than :data:`TEXT_MAX_SEQUENCE_LENGTH` tokens raises: the pipeline would truncate it silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from distill_to_loop.data.text import ATTN_MASK, POOLED, TEXT_EMB

TEXT_MAX_SEQUENCE_LENGTH = 512  # ZImagePipeline's max_sequence_length: the tokens the pipeline keeps at most
STORED_TEXT_ROWS = 128  # rows stored per caption in the training cache (the chat-templated COCO train captions take 15 to 72 tokens)


class ZImagePromptEncoder:
    """Qwen3 prompt encoding through ``ZImagePipeline._encode_prompt`` (module docstring). ``pipe`` is a ``ZImagePipeline`` holding at least the
    tokenizer and the text encoder (:meth:`from_pretrained` loads only those two).

    ``__call__`` returns the training-cache form: ``text_emb`` ``[n, stored_rows, D]`` (zero rows after a caption's length), ``pooled`` the
    validity mask ``[n, stored_rows]`` (1.0 for a real token, in the text dtype) and ``attn_mask`` the same mask as bool. :meth:`rows` returns the
    pipeline's list of ``[len_i, D]`` rows."""

    def __init__(self, pipe, stored_rows: int = STORED_TEXT_ROWS, max_sequence_length: int = TEXT_MAX_SEQUENCE_LENGTH,
                 device: str | torch.device = "cpu") -> None:
        self.pipe = pipe
        self.stored_rows = int(stored_rows)
        self.max_sequence_length = int(max_sequence_length)
        self.device = torch.device(device)
        pipe.text_encoder.eval()

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device: str | torch.device, dtype: torch.dtype = torch.bfloat16, **kwargs) -> "ZImagePromptEncoder":
        """The tokenizer and the text encoder of the local teacher directory ``model_dir`` (no transformer, no VAE)."""
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(str(model_dir), transformer=None, vae=None, dtype=dtype, local_files_only=True).to(device)
        return cls(pipe, device=device, **kwargs)

    def token_counts(self, captions: Sequence[str]) -> list[int]:
        """The untruncated token count of each caption in the chat template of ``ZImagePipeline._encode_prompt`` (the same
        ``apply_chat_template`` arguments and tokenizer call, without the truncation and padding)."""
        tokenizer = self.pipe.tokenizer
        templated = [
            tokenizer.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for c in captions
        ]
        return [len(ids) for ids in tokenizer(templated).input_ids]

    def rows(self, captions: Sequence[str]) -> list[torch.Tensor]:
        """One ``[len_i, D]`` tensor per caption; ``ValueError`` for a caption the pipeline would truncate (more than ``max_sequence_length``
        tokens; counted with :meth:`token_counts` only for a caption that fills ``max_sequence_length`` rows)."""
        with torch.no_grad():
            rows = self.pipe._encode_prompt(prompt=list(captions), device=self.device, max_sequence_length=self.max_sequence_length)  # a copy: the
            # pipeline rewrites the list it is given in place
        at_limit = [i for i, r in enumerate(rows) if r.shape[0] >= self.max_sequence_length]
        if at_limit:
            truncated = [i for i, n in zip(at_limit, self.token_counts([captions[i] for i in at_limit])) if n > self.max_sequence_length]
            if truncated:
                raise ValueError(f"captions {truncated} have more than {self.max_sequence_length} tokens, which the pipeline truncates; shorten them")
        return rows

    def __call__(self, captions: Sequence[str]) -> dict[str, torch.Tensor]:
        rows = self.rows(captions)
        too_long = [i for i, r in enumerate(rows) if r.shape[0] > self.stored_rows]
        if too_long:
            raise ValueError(f"captions {too_long} have more than {self.stored_rows} tokens; filter them out before encoding")
        text = rows[0].new_zeros(len(rows), self.stored_rows, rows[0].shape[-1])
        mask = torch.zeros(len(rows), self.stored_rows, dtype=torch.bool)
        for i, r in enumerate(rows):
            text[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = True
        return {TEXT_EMB: text, POOLED: mask.to(text.dtype), ATTN_MASK: mask}
