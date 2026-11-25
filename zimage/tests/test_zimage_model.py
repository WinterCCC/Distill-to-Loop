"""The looped Z-Image student and the teacher adapter on tiny random models (CPU, fp32)."""

from __future__ import annotations

import diffusers
import pytest
import torch

import zimage_tiny as tiny
from distill_to_loop.models.looped_sd3 import stride_loop_sources
from dooop_zimage.model import LoopedZImageTransformer, ZImageTeacherAdapter, caption_rows, image_list, model_time


def _inputs(batch: int = 2, seed: int = 1):
    """``(latents [B, C, H, W], sigma [B], text [B, L, D], mask [B, L])`` with unequal caption lengths."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, tiny.C, tiny.H, tiny.H, generator=g)
    sigma = torch.rand(batch, generator=g) * 0.9 + 0.05
    lengths = [5, 40, 33][:batch]
    mask = torch.zeros(batch, tiny.TEXT_ROWS)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
    text = torch.randn(batch, tiny.TEXT_ROWS, tiny.D, generator=g) * mask[..., None]
    return x, sigma, text, mask


def test_released_split_initializes_from_every_fourth_middle_layer():
    assert stride_loop_sources(30, 3, 6, 3) == [3, 7, 11, 15, 19, 23]


def test_from_teacher_copies_the_teacher_layers():
    teacher = tiny.tiny_transformer()
    student = LoopedZImageTransformer.from_teacher(teacher, n_pre=3, n_loop=3, n_post=3, n_rounds=2)  # M = 6, shared = layers 3, 5, 7
    state, teacher_state = student.state_dict(), teacher.state_dict()
    for blocks, layers in (("pre", [0, 1, 2]), ("loop", [3, 5, 7]), ("post", [9, 10, 11])):
        for j, layer in enumerate(layers):
            for key, value in teacher.layers[layer].state_dict().items():
                assert torch.equal(state[f"{blocks}.{j}.{key}"], value), (blocks, j, key)
    shared = [k for k in state if not k.startswith(("pre.", "loop.", "post.", "depth_embedder."))]
    assert shared and all(torch.equal(state[k], teacher_state[k]) for k in shared)
    assert torch.count_nonzero(student.depth_embedder.embedder.linear_2.weight) == 0  # e_s is zero at initialization
    assert student.config.n_layers == teacher.config.n_layers and student.loop_rounds == 2


def test_one_round_of_the_full_depth_reproduces_the_teacher():
    """3 + 6 + 3 blocks of the 12-layer teacher with K = 1: every teacher layer once, the zero depth embedding, alpha = 1."""
    teacher = tiny.tiny_transformer()
    student = LoopedZImageTransformer.from_teacher(teacher, n_pre=3, n_loop=6, n_post=3, n_rounds=1).eval()
    x, sigma, text, mask = _inputs()
    with torch.no_grad():
        out_student = student(image_list(x), 1.0 - sigma, caption_rows(text, mask), return_dict=False)[0]
        out_teacher = teacher(image_list(x), 1.0 - sigma, caption_rows(text, mask), return_dict=False)[0]
    for a, b in zip(out_student, out_teacher):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rounds", [1, 2, 4])
def test_forward_rounds_and_the_pipeline_forward_agree(rounds):
    student = LoopedZImageTransformer.from_teacher(tiny.tiny_transformer(), n_rounds=4).eval()
    with torch.no_grad():
        for p in student.depth_embedder.parameters():  # a non-zero depth embedding, so the rounds differ
            p.add_(0.1)
        x, sigma, text, mask = _inputs()
        student.loop_rounds = rounds
        raw = student(image_list(x), model_time(sigma * 1000.0), caption_rows(text, mask), return_dict=False)[0]
        last = student.forward_rounds(x, sigma * 1000.0, text, mask, n_rounds=rounds, readout="last")
        every = student.forward_rounds(x, sigma * 1000.0, text, mask, n_rounds=rounds, readout="all", include_prelude=True)
    assert len(last) == 1 and len(every) == rounds + 1
    assert torch.equal(last[0], -torch.stack(raw).squeeze(2))
    assert torch.equal(every[-1], last[0])


def test_given_uniform_depth_grid_matches_the_default_grid():
    student = LoopedZImageTransformer.from_teacher(tiny.tiny_transformer(), n_rounds=4).eval()
    x, sigma, text, mask = _inputs()
    grid = torch.arange(1, 5, dtype=torch.float32).div(4).expand(x.shape[0], 4)
    with torch.no_grad():
        a = student.forward_rounds(x, sigma * 1000.0, text, mask, readout="all")
        b = student.forward_rounds(x, sigma * 1000.0, text, mask, readout="all", depth_s=grid)
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v, rtol=0, atol=1e-6)


def test_activation_checkpointing_gives_the_same_gradients():
    x, sigma, text, mask = _inputs()
    depth = torch.tensor([[0.2, 0.5, 0.7, 1.0], [0.1, 0.3, 0.9, 1.0]])
    grads = []
    for loop_ckpt, block_ckpt in ((False, False), (True, False), (False, True), (True, True)):
        torch.manual_seed(0)  # the same random first layer of the depth embedding
        student = LoopedZImageTransformer.from_teacher(tiny.tiny_transformer(), n_rounds=4)
        student.loop_grad_checkpoint = loop_ckpt
        if block_ckpt:
            student.enable_gradient_checkpointing()
        outs = student.forward_rounds(x, sigma * 1000.0, text, mask, depth_s=depth, include_prelude=True)
        torch.stack([o.square().mean() for o in outs]).sum().backward()
        grads.append([p.grad.clone() for p in student.parameters()])
    for other in grads[1:]:
        assert all(torch.equal(a, b) for a, b in zip(grads[0], other))


def test_argument_checks():
    student = LoopedZImageTransformer.from_teacher(tiny.tiny_transformer(), n_rounds=2).eval()
    x, sigma, text, mask = _inputs()
    with pytest.raises(ValueError, match="depth_s must have shape"):
        student.forward_rounds(x, sigma, text, mask, depth_s=torch.ones(2, 3))
    with pytest.raises(ValueError, match="readout"):
        student.forward_rounds(x, sigma, text, mask, readout="first")
    with pytest.raises(ValueError, match="no valid token"):
        student.forward_rounds(x, sigma, text, torch.zeros_like(mask))
    with pytest.raises(NotImplementedError):
        student(image_list(x), sigma, caption_rows(text, mask), siglip_feats=[[x[0]]])


def test_another_diffusers_version_is_refused(monkeypatch):
    monkeypatch.setattr(diffusers, "__version__", "0.41.0")
    with pytest.raises(RuntimeError, match="diffusers==0.40.0"):
        LoopedZImageTransformer(**tiny.TINY_TEACHER)


def test_teacher_adapter_is_the_negated_pipeline_call():
    teacher = tiny.tiny_transformer()
    adapter = ZImageTeacherAdapter(teacher, compute_dtype=torch.float32)
    x, sigma, text, mask = _inputs()
    with torch.no_grad():
        (velocity,) = adapter(x, sigma * 1000.0, text, mask)
        raw = teacher(image_list(x), (1000.0 - sigma * 1000.0) / 1000.0, caption_rows(text, mask), return_dict=False)[0]
    assert adapter.dtype == torch.float32 and velocity.shape == x.shape
    assert torch.equal(velocity, -torch.stack(raw).squeeze(2))
