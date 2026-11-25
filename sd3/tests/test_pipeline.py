"""The pipeline ``load_pipeline`` returns and the image-size check, on tiny components (CPU)."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch
from diffusers import AutoencoderKL

import dooop.pipeline as pipeline_module
from dooop.inference import build_sampling_pipeline, latent_shape
from dooop.models.looped_sd3 import LoopedSD3Transformer
from dooop.pipeline import LoopedSD3Pipeline
from tests import tiny


def _components():
    torch.manual_seed(0)
    student = LoopedSD3Transformer.from_teacher(tiny.tiny_teacher(), n_pre=2, n_loop=5, n_post=2, n_rounds=4).float().eval()
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(8, 8),
        layers_per_block=1, latent_channels=16, norm_num_groups=4, sample_size=16, scaling_factor=1.5305, shift_factor=0.0609,
    ).eval()
    return student, vae


def _looped_pipeline(student, vae) -> LoopedSD3Pipeline:
    pipe = LoopedSD3Pipeline(
        transformer=student, scheduler=tiny.tiny_scheduler(), vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
        text_encoder_3=None, tokenizer_3=None,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _encodings(n: int = 2):
    g = torch.Generator().manual_seed(0)
    return dict(
        prompt_embeds=torch.randn(n, tiny.SEQ, tiny.D, generator=g), pooled_prompt_embeds=torch.randn(n, tiny.D_POOLED, generator=g),
        negative_prompt_embeds=torch.randn(n, tiny.SEQ, tiny.D, generator=g), negative_pooled_prompt_embeds=torch.randn(n, tiny.D_POOLED, generator=g),
    )


def test_on_the_cpu_the_call_is_that_of_the_diffusers_pipeline():
    student, vae = _components()
    kwargs = dict(**_encodings(), height=16, width=16, num_inference_steps=2, guidance_scale=4.0, output_type="latent")
    expected = build_sampling_pipeline(vae, tiny.tiny_scheduler(), student)(**kwargs, generator=torch.Generator().manual_seed(1)).images
    got = _looped_pipeline(student, vae)(**kwargs, generator=torch.Generator().manual_seed(1)).images
    assert torch.equal(expected, got)


def test_trained_text_length_and_autocast_device_are_the_defaults(monkeypatch):
    student, vae = _components()
    pipe = _looped_pipeline(student, vae)
    lengths, autocast_devices = [], []
    encodings = _encodings(1)

    def encode_prompt(*args, **kwargs):  # stands in for the text towers
        lengths.append(kwargs["max_sequence_length"])
        return encodings["prompt_embeds"], encodings["negative_prompt_embeds"], encodings["pooled_prompt_embeds"], encodings["negative_pooled_prompt_embeds"]

    def autocast(device):
        autocast_devices.append(torch.device(device))
        return contextlib.nullcontext()

    pipe.encode_prompt = encode_prompt
    monkeypatch.setattr(pipeline_module, "student_autocast", autocast)
    common = dict(height=16, width=16, num_inference_steps=1, guidance_scale=4.0, output_type="latent")
    pipe("a prompt", **common)
    pipe("a prompt", max_sequence_length=128, **common)
    assert lengths == [77, 128]
    assert autocast_devices == [pipe._execution_device] * 2 == [torch.device("cpu")] * 2  # the call reads the device the pipeline runs on


def test_autocast_covers_the_transformer_forward_only(monkeypatch):
    """The CUDA path on the CPU: ``student_autocast`` replaced by a context that marks where it is open. Every transformer forward runs inside
    it; the text encoding and the VAE decode run outside it."""
    student, vae = _components()
    pipe = _looped_pipeline(student, vae)
    inside, seen = [False], []
    encodings = _encodings(1)

    @contextlib.contextmanager
    def autocast(device):
        inside[0] = True
        try:
            yield
        finally:
            inside[0] = False

    def encode_prompt(*args, **kwargs):  # stands in for the text towers
        seen.append(("text", inside[0]))
        return encodings["prompt_embeds"], encodings["negative_prompt_embeds"], encodings["pooled_prompt_embeds"], encodings["negative_pooled_prompt_embeds"]

    forward, decode = student.forward, vae.decode

    def probe_forward(*args, **kwargs):
        seen.append(("transformer", inside[0]))
        return forward(*args, **kwargs)

    def probe_decode(*args, **kwargs):
        seen.append(("vae", inside[0]))
        return decode(*args, **kwargs)

    pipe.encode_prompt = encode_prompt
    monkeypatch.setattr(student, "forward", probe_forward)
    monkeypatch.setattr(vae, "decode", probe_decode)
    monkeypatch.setattr(pipeline_module, "student_autocast", autocast)
    pipe("a prompt", height=16, width=16, num_inference_steps=2, guidance_scale=4.0, output_type="pt")
    assert seen == [("text", False), ("transformer", True), ("transformer", True), ("vae", False)]
    assert student.forward is probe_forward  # the call leaves the transformer as it found it


def test_image_size_must_be_a_multiple_of_the_vae_factor_times_the_patch_size():
    student, vae = _components()  # VAE factor 2 here (two blocks) x patch size 2
    assert latent_shape(student, vae, 16) == (16, 8, 8)
    for bad in (6, 0):
        with pytest.raises(ValueError, match="multiple of 4"):
            latent_shape(student, vae, bad)
    sd3_vae = types.SimpleNamespace(config=types.SimpleNamespace(block_out_channels=[128, 256, 512, 512]))  # SD3-medium's VAE: factor 8
    assert latent_shape(student, sd3_vae, 512) == (16, 64, 64)
    with pytest.raises(ValueError, match="multiple of 16"):
        latent_shape(student, sd3_vae, 520)
