"""Distill-to-Loop: distill SD3-medium into a looped, parameter-shared student whose loop count K is chosen at sampling time.

Public API::

    from distill_to_loop import load_student, load_pipeline, encode_prompts, generate_images, export_student, LoopedSD3Transformer
"""

from distill_to_loop.inference import encode_prompts, export_student, generate_images, load_pipeline, load_student, load_vae_and_scheduler_factory
from distill_to_loop.models.looped_sd3 import LoopedSD3Transformer

__all__ = [
    "LoopedSD3Transformer",
    "encode_prompts",
    "export_student",
    "generate_images",
    "load_pipeline",
    "load_student",
    "load_vae_and_scheduler_factory",
]
