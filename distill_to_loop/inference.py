"""Export, loading and sampling of the student.

An exported student is a directory in the diffusers layout: ``config.json`` (the student's architecture) and
``diffusion_pytorch_model.safetensors`` (the EMA weights in fp32, with the teacher model id and the training step in the file's metadata). It
loads without the teacher transformer (:func:`load_student`) and samples inside a ``StableDiffusion3Pipeline`` built from the teacher's VAE,
scheduler and, for text prompts, text encoders (:func:`load_pipeline`). The loop count ``K`` is the student's ``loop_rounds``.

Sampling follows the evaluation protocol of the released model: FlowMatch Euler on the scheduler's ``steps``-point grid, CFG with the empty
prompt as the negative, the initial noise of image ``(prompt p, sample s)`` a function of ``(seed, p, s)`` only
(:func:`distill_to_loop.seeding.image_noise`), every prompt encoded once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from distill_to_loop.hub import explain_missing_model
from distill_to_loop.models.looped_sd3 import LoopedSD3Transformer
from distill_to_loop.precision import student_autocast, text_dtype_for, vae_dtype_for
from distill_to_loop.seeding import image_noise

WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
CONFIG_NAME = "config.json"
METADATA_KEY = "distill_to_loop"  # the weight file's metadata entry
# Student options of the training config that the released architecture fixes; a recorded config holding another value is refused.
FIXED_MODEL_FIELDS: dict[str, Any] = {
    "untie_loop": False, "loop_index_cond": False, "guidance_embed": False, "depth_embed": True, "depth_cond": True,
    "loop_step_scale": "delta_s_unit", "qk_norm": "loop_post", "loop_attn_mod": "xsa_soft", "family": "sd3",
}
DEFAULT_LOOP_ROUNDS, DEFAULT_STEPS, DEFAULT_CFG_SCALE, DEFAULT_RESOLUTION = 4, 28, 7.0, 512
TEXT_MAX_SEQUENCE_LENGTH = 77


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def check_recorded_model(model_cfg: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a checkpoint's recorded ``model`` section sets one of :data:`FIXED_MODEL_FIELDS` to another value (a field it
    does not record is accepted)."""
    wrong = {k: (model_cfg[k], v) for k, v in FIXED_MODEL_FIELDS.items() if k in model_cfg and model_cfg[k] != v}
    if wrong:
        raise ValueError(f"the checkpoint's student differs from the released architecture (recorded, expected): {wrong}")


def teacher_transformer_config(model_cfg: dict[str, Any], hf_cache: str | None = None) -> dict[str, Any]:
    """The teacher transformer's config (``config.json``, the constructor defaults filled in for absent keys): from ``teacher_transformer_path``
    when the record sets one, else the ``transformer`` subfolder of ``teacher_model_id`` in the local Hugging Face cache."""
    import inspect

    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    path = model_cfg.get("teacher_transformer_path") or ""
    if path:
        stored = dict(SD3Transformer2DModel.load_config(path, local_files_only=True))
    else:
        with explain_missing_model(model_cfg["teacher_model_id"], hf_cache):
            stored = dict(SD3Transformer2DModel.load_config(model_cfg["teacher_model_id"], subfolder="transformer", cache_dir=hf_cache, local_files_only=True))
    defaults = {name: p.default for name, p in inspect.signature(SD3Transformer2DModel.__init__).parameters.items() if p.default is not inspect.Parameter.empty}
    return {**defaults, **stored}


def student_architecture(model_cfg: dict[str, Any], teacher_cfg: dict[str, Any]) -> dict[str, Any]:
    """The :class:`LoopedSD3Transformer` constructor arguments of a recorded training ``model`` section on a teacher config."""
    keys = ("sample_size", "patch_size", "in_channels", "attention_head_dim", "num_attention_heads", "joint_attention_dim", "caption_projection_dim",
            "pooled_projection_dim", "out_channels", "pos_embed_max_size", "qk_norm")
    arch = {k: teacher_cfg[k] for k in keys}
    arch.update({k: int(model_cfg[k]) for k in ("n_pre", "n_loop", "n_post", "n_rounds")})
    return arch


def export_student(checkpoint: str | Path, out_dir: str | Path, hf_cache: str | None = None) -> Path:
    """Write the EMA weights of a training checkpoint as an exported student into ``out_dir`` and return the weight file. The checkpoint is
    memory-mapped and never written. The weights are loaded into a student of the recorded architecture with ``strict=True`` before they are
    written."""
    from distill_to_loop.training.checkpoint import EMA_CHECKPOINT_KEY, read_checkpoint_payload

    payload = read_checkpoint_payload(checkpoint, map_location="cpu", mmap=True)
    extra = payload.get("extra", {})
    if EMA_CHECKPOINT_KEY not in extra:
        raise ValueError(f"{checkpoint}: the checkpoint records no EMA weights")
    model_cfg = dict(extra["config"]["model"])
    check_recorded_model(model_cfg)
    arch = student_architecture(model_cfg, teacher_transformer_config(model_cfg, hf_cache))
    state = {k: v.detach().to(torch.float32).contiguous() for k, v in extra[EMA_CHECKPOINT_KEY]["state_dict"].items()}
    with torch.device("meta"):
        student = LoopedSD3Transformer(**arch)
    student.load_state_dict(state, strict=True, assign=True)  # the strict key and shape check
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_config(out_dir)
    metadata = {
        "teacher_model_id": str(model_cfg["teacher_model_id"]), "training_step": str(int(payload["step"])),
        "ema_decay": str(extra[EMA_CHECKPOINT_KEY]["decay"]), "weights": "ema",
    }
    path = out_dir / WEIGHTS_NAME
    # One metadata entry holding sorted JSON: the file header (and so the file's bytes) is then the same on every export of a checkpoint.
    save_file(state, str(path), metadata={METADATA_KEY: json.dumps(metadata, sort_keys=True)})
    return path


def read_student_metadata(student_dir: str | Path) -> dict[str, str]:
    """The metadata of an exported student's weight file (``teacher_model_id``, ``training_step``, ``ema_decay``, ``weights``)."""
    with safe_open(str(Path(student_dir) / WEIGHTS_NAME), framework="pt") as f:
        return json.loads((f.metadata() or {})[METADATA_KEY])


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_student(student_dir: str | Path, device: str | torch.device = "cpu", loop_rounds: int = DEFAULT_LOOP_ROUNDS) -> LoopedSD3Transformer:
    """An exported student in fp32 on ``device``, in eval mode, running ``loop_rounds`` rounds (``K``; change it later with
    ``student.loop_rounds = K``). The weights are loaded with ``strict=True``."""
    student_dir = Path(student_dir)
    config = LoopedSD3Transformer.load_config(student_dir)
    with torch.device("meta"):
        student = LoopedSD3Transformer.from_config(config)
    # Fresh storage, then a copy: tensors backed by the file's buffer are not 64-byte aligned, and CPU kernels on unaligned weights may sum in
    # another order (outputs then differ in the last bits). Every tensor of the student is in the state dict (no non-persistent buffer).
    # ``float()`` rather than ``to(dtype=...)``: diffusers' ``ModelMixin.to`` logs a warning about its (here empty) list of float32 modules.
    student = student.to_empty(device=device).float()
    student.load_state_dict(load_file(str(student_dir / WEIGHTS_NAME)), strict=True)
    student.loop_rounds = int(loop_rounds)
    student.eval()
    return student


def load_vae_and_scheduler_factory(model_id: str, device: str | torch.device, cache_dir: str | None = None):
    """``(vae, scheduler_factory)`` of the teacher model: the VAE (bf16 on CUDA, fp32 elsewhere) and a function returning a fresh copy of its
    FlowMatch Euler scheduler."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

    with explain_missing_model(model_id, cache_dir):
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", cache_dir=cache_dir, local_files_only=True, torch_dtype=vae_dtype_for(device)).to(device)

    def scheduler_factory():
        return FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir=cache_dir, local_files_only=True)

    return vae, scheduler_factory


def build_sampling_pipeline(vae: Any, scheduler: Any, transformer: Any) -> Any:
    """A ``StableDiffusion3Pipeline`` without text towers (the prompt encodings are passed in)."""
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline(
        transformer=transformer, scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
        text_encoder_3=None, tokenizer_3=None,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_pipeline(student_dir: str | Path, device: str | torch.device = "cuda", model_id: str | None = None, cache_dir: str | None = None,
                  loop_rounds: int = DEFAULT_LOOP_ROUNDS) -> Any:
    """A ready :class:`~distill_to_loop.pipeline.LoopedSD3Pipeline` (a ``StableDiffusion3Pipeline``) with the exported student (fp32) as its
    transformer and the teacher model's text encoders, VAE and scheduler (bf16 on CUDA, fp32 elsewhere; ``model_id`` defaults to the teacher
    recorded in the export). Its call uses the trained T5 length (``max_sequence_length=77``) and, on CUDA, bf16 autocast around the
    transformer by default. Set the loop count with ``pipe.transformer.loop_rounds = K``."""
    from distill_to_loop.pipeline import LoopedSD3Pipeline

    model_id = model_id or read_student_metadata(student_dir)["teacher_model_id"]
    student = load_student(student_dir, device=device, loop_rounds=loop_rounds)
    with explain_missing_model(model_id, cache_dir):
        pipe = LoopedSD3Pipeline.from_pretrained(
            model_id, transformer=student, cache_dir=cache_dir, local_files_only=True, dtype=vae_dtype_for(device),
        )
    return pipe.to(device)


# --------------------------------------------------------------------------- #
# sampling with precomputed prompt encodings
# --------------------------------------------------------------------------- #
def encode_prompts(texts: Sequence[str], model_id: str, device: str | torch.device, cache_dir: str | None = None) -> dict[str, torch.Tensor]:
    """The bf16 encodings of ``texts`` and of the empty prompt, from one call of the three text towers (which are then released):
    ``prompt_embeds [N, 154, 4096]``, ``pooled [N, 2048]``, ``negative_prompt_embeds [154, 4096]``, ``negative_pooled [2048]`` on the CPU. The
    prompts are not split into chunks: a prompt's encoding can change in the last bit with the batch it is encoded in."""
    from distill_to_loop.data.text import POOLED, TEXT_EMB, Sd3PromptEncoder

    encoder = Sd3PromptEncoder(model_id, device=device, dtype=torch.bfloat16, max_sequence_length=TEXT_MAX_SEQUENCE_LENGTH, cache_dir=cache_dir)
    out = encoder([*texts, ""])
    embeds, pooled = out[TEXT_EMB].cpu(), out[POOLED].cpu()
    del encoder, out
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return {"prompt_embeds": embeds[:-1], "pooled": pooled[:-1], "negative_prompt_embeds": embeds[-1], "negative_pooled": pooled[-1]}


def latent_shape(student: LoopedSD3Transformer, vae: Any, resolution: int) -> tuple[int, int, int]:
    """The ``(C, h, w)`` initial-noise shape of a square image of ``resolution`` pixels, which must be a multiple of the VAE's downsampling
    factor times the student's patch size (8 x 2 = 16 for SD3-medium)."""
    factor = 2 ** (len(vae.config.block_out_channels) - 1)
    multiple = factor * int(student.config.patch_size)
    if resolution < 1 or resolution % multiple:
        raise ValueError(f"the image size must be a positive multiple of {multiple} (VAE factor {factor} x patch size), got {resolution}")
    return (int(student.config.in_channels), resolution // factor, resolution // factor)


@torch.no_grad()
def sample_with_encodings(
    vae: Any,
    scheduler_factory: Any,
    model: Any,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_pooled: torch.Tensor,
    noise: torch.Tensor,
    steps: int,
    cfg_scale: float,
    height: int,
    width: int,
    device: torch.device | str,
    micro_batch_size: int | None = None,
    output_type: str = "pil",
) -> list:
    """One pipeline call per micro-batch on the given encodings (rows aligned) and initial ``noise`` (``latents=``, cast to the encodings'
    dtype): FlowMatch Euler with ``steps`` steps and CFG ``cfg_scale`` (at most 1: the conditional branch only), the model under
    :func:`~distill_to_loop.precision.student_autocast`. Returns PIL images, or latents with ``output_type="latent"``."""
    n = prompt_embeds.shape[0]
    mb = n if micro_batch_size is None else min(micro_batch_size, n)
    results: list = []
    for start in range(0, n, mb):
        end = min(start + mb, n)
        pipe = build_sampling_pipeline(vae, scheduler_factory(), model)
        with student_autocast(device):
            result = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds[start:end],
                negative_prompt_embeds=negative_prompt_embeds[start:end],
                pooled_prompt_embeds=pooled[start:end],
                negative_pooled_prompt_embeds=negative_pooled[start:end],
                latents=noise[start:end].to(device, dtype=prompt_embeds.dtype),
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=float(cfg_scale),
                output_type=output_type,
            )
        results.extend(result.images if output_type == "pil" else list(result.images))
    return results


def generate_images(
    student: LoopedSD3Transformer,
    vae: Any,
    scheduler_factory: Any,
    encodings: dict[str, torch.Tensor],
    *,
    loop_rounds: int = DEFAULT_LOOP_ROUNDS,
    steps: int = DEFAULT_STEPS,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    resolution: int = DEFAULT_RESOLUTION,
    seed: int = 0,
    num_samples: int = 1,
    micro_batch_size: int = 4,
    device: str | torch.device = "cuda",
    output_type: str = "pil",
) -> list[tuple[int, int, Any]]:
    """``[(prompt index, sample index, image)]`` for every prompt of ``encodings`` (:func:`encode_prompts`) and sample, prompt-major. Image
    ``(p, s)`` starts from :func:`~distill_to_loop.seeding.image_noise` of ``(seed, p, s)``. Every pipeline call has exactly
    ``micro_batch_size`` rows (a short last call is padded with copies of its last row, whose images are dropped), so the batch shape does not
    depend on the number of prompts."""
    student.loop_rounds = int(loop_rounds)
    shape = latent_shape(student, vae, resolution)
    dtype = text_dtype_for(device)
    pairs = [(p, s) for p in range(encodings["prompt_embeds"].shape[0]) for s in range(num_samples)]
    out: list[tuple[int, int, Any]] = []
    for start in range(0, len(pairs), micro_batch_size):
        chunk = pairs[start : start + micro_batch_size]
        rows = chunk + [chunk[-1]] * (micro_batch_size - len(chunk))
        index = torch.as_tensor([p for p, _ in rows], dtype=torch.long)
        noise = torch.stack([image_noise(seed, p, s, shape) for p, s in rows])
        negative_embeds = encodings["negative_prompt_embeds"].unsqueeze(0).expand(len(rows), *encodings["negative_prompt_embeds"].shape)
        negative_pooled = encodings["negative_pooled"].unsqueeze(0).expand(len(rows), *encodings["negative_pooled"].shape)
        images = sample_with_encodings(
            vae, scheduler_factory, student,
            encodings["prompt_embeds"][index].to(device, dtype=dtype), encodings["pooled"][index].to(device, dtype=dtype),
            negative_embeds.to(device, dtype=dtype), negative_pooled.to(device, dtype=dtype),
            noise, steps, cfg_scale, resolution, resolution, device, output_type=output_type,
        )
        out.extend((p, s, image) for (p, s), image in zip(chunk, images))
    return out


def write_generation_record(out_dir: str | Path, record: dict[str, Any]) -> Path:
    path = Path(out_dir) / "generation.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
