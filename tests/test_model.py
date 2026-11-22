"""The looped student and its attention options, on tiny random components."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0

from distill_to_loop.models.attention import (
    QK_NORM_IMG,
    XSA_BETA,
    LoopedJointAttnProcessor,
    exclusive_self_attention,
    install_joint_attention_options,
)
from distill_to_loop.models.looped_sd3 import LoopedSD3Transformer, loop_step_weights, stride_loop_sources
from tests import tiny


def _attention(seed: int = 0) -> Attention:
    torch.manual_seed(seed)
    return Attention(query_dim=16, cross_attention_dim=None, added_kv_proj_dim=16, dim_head=8, heads=2, out_dim=16, context_pre_only=False,
                     bias=True, processor=JointAttnProcessor2_0())


def test_processor_without_options_equals_diffusers():
    attn = _attention()
    h, c = torch.randn(2, 5, 16), torch.randn(2, 3, 16)
    expected = attn(h, c)
    attn.set_processor(LoopedJointAttnProcessor())
    got = attn(h, c)
    assert all(torch.equal(a, b) for a, b in zip(expected, got))


def test_options_add_parameters_and_are_identity_at_zero_beta():
    attn = _attention()
    h, c = torch.randn(2, 5, 16), torch.randn(2, 3, 16)
    expected = attn(h, c)
    install_joint_attention_options(attn, qk_norm=False, xsa_soft=True)
    assert torch.equal(getattr(attn, XSA_BETA), torch.zeros(2))
    got = attn(h, c)  # beta = 0: soft XSA removes nothing
    assert all(torch.equal(a, b) for a, b in zip(expected, got))
    install_joint_attention_options(_attention(), qk_norm=True, xsa_soft=False)


def test_xsa_removes_the_own_value_component():
    out, v = torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)
    full = exclusive_self_attention(out, v)
    assert torch.allclose((full * torch.nn.functional.normalize(v, dim=-1)).sum(-1), torch.zeros(1, 2, 4), atol=1e-6)
    assert torch.equal(exclusive_self_attention(out, v, torch.ones(2)), full)


def test_stride_sources_and_teacher_copy():
    assert stride_loop_sources(24, 2, 5, 2) == [2, 6, 10, 14, 18]
    with pytest.raises(ValueError):
        stride_loop_sources(23, 2, 5, 2)
    teacher = tiny.tiny_teacher(num_layers=14)
    student = LoopedSD3Transformer.from_teacher(teacher, n_pre=2, n_loop=5, n_post=2, n_rounds=4)
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    for j, layer in enumerate([2, 4, 6, 8, 10]):
        assert torch.equal(s_state[f"loop.{j}.attn.to_q.weight"], t_state[f"transformer_blocks.{layer}.attn.to_q.weight"].float())
    assert torch.equal(s_state["post.1.attn.to_q.weight"], t_state["transformer_blocks.13.attn.to_q.weight"].float())
    assert torch.equal(s_state["depth_embedder.embedder.linear_2.weight"], torch.zeros_like(s_state["depth_embedder.embedder.linear_2.weight"]))
    assert hasattr(student.loop[0].attn, QK_NORM_IMG) and hasattr(student.post[0].attn, QK_NORM_IMG) and not hasattr(student.pre[0].attn, QK_NORM_IMG)
    assert hasattr(student.loop[0].attn, XSA_BETA) and not hasattr(student.post[0].attn, XSA_BETA)


def _student() -> LoopedSD3Transformer:
    return LoopedSD3Transformer.from_teacher(tiny.tiny_teacher(), n_pre=2, n_loop=5, n_post=2, n_rounds=4).float().eval()


def _inputs(batch: int = 2):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(batch, tiny.C, tiny.H, tiny.H, generator=g), torch.tensor([300.0, 800.0][:batch]),
            torch.randn(batch, tiny.SEQ, tiny.D, generator=g), torch.randn(batch, tiny.D_POOLED, generator=g))


@pytest.mark.parametrize("rounds", [1, 2, 4, 6])
def test_forward_runs_any_loop_count(rounds):
    student = _student()
    x, t, e, p = _inputs()
    with torch.no_grad():
        outs = student.forward_rounds(x, t, e, p, n_rounds=rounds, readout="all", include_prelude=True)
        student.loop_rounds = rounds
        last = student(x, e, p, t).sample
    assert len(outs) == rounds + 1 and all(o.shape == x.shape for o in outs)
    assert torch.equal(outs[-1], last)


def test_uniform_grid_steps_and_given_coordinates():
    assert loop_step_weights(None, 4) == [0.25] * 4
    depth = torch.tensor([[0.1, 0.5, 1.0]])
    assert [w.item() for w in loop_step_weights(depth, 3)] == pytest.approx([0.1, 0.4, 0.5])
    student = _student()
    x, t, e, p = _inputs(1)
    with torch.no_grad():
        grid = student.forward_rounds(x, t, e, p, n_rounds=4, readout="last")[0]
        given = student.forward_rounds(x, t, e, p, n_rounds=4, readout="last", depth_s=torch.tensor([[0.25, 0.5, 0.75, 1.0]]))[0]
    assert torch.allclose(grid, given, atol=1e-5)
    with pytest.raises(ValueError):
        student.forward_rounds(x, t, e, p, n_rounds=4, depth_s=torch.ones(1, 3))


@pytest.mark.parametrize("round_checkpointing, block_checkpointing", [(False, True), (True, False), (True, True)])
def test_activation_checkpointing_keeps_outputs_and_gradients(round_checkpointing, block_checkpointing):
    """Round-level and per-block activation checkpointing, alone or nested, give the outputs, loss and gradients of a plain forward bit for
    bit; the per-block path really runs."""
    x, t, e, p = _inputs()
    results = []
    for use_round, use_block in ((False, False), (round_checkpointing, block_checkpointing)):
        torch.manual_seed(0)  # the same random initialization of the depth embedder's first layer
        student = _student().train()
        student.loop_grad_checkpoint = use_round
        block_calls = []
        if use_block:
            student.enable_gradient_checkpointing()
            inner = student._gradient_checkpointing_func

            def counting(*args, _inner=inner):
                block_calls.append(1)
                return _inner(*args)

            student._gradient_checkpointing_func = counting
        outs = student.forward_rounds(x, t, e, p, n_rounds=3, readout="all", depth_s=torch.tensor([[0.2, 0.7, 1.0], [0.4, 0.5, 1.0]]), include_prelude=True)
        loss = sum(o.square().mean() for o in outs)
        loss.backward()
        assert (len(block_calls) > 0) == use_block
        results.append((loss.detach(), [o.detach() for o in outs], [q.grad.clone() for q in student.parameters()]))
    (loss_a, outs_a, grads_a), (loss_b, outs_b, grads_b) = results
    assert torch.equal(loss_a, loss_b)
    assert all(torch.equal(a, b) for a, b in zip(outs_a, outs_b))
    assert len(grads_a) == len(grads_b) and all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))
