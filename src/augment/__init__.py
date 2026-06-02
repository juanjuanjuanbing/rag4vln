# -*- coding: utf-8 -*-
"""
Incomplete instruction augmentation: retrieval evidence + short intent → executable navigation instruction.

- ``InstructionAugmenter``: base class.
- ``LLMDirectInstructionAugmenter``: type 1, single LLM generation.
- ``TemplatePathInstructionAugmenter``: type 2, LLM slot filling + local template.
- ``SemanticPathPlanningAugmenter``: type 3, semantic path planning (three-stage CoT + final VLN instruction).
- ``ROnlyInstructionAugmenter``: R_only baseline, concatenates evidence and original instruction (no LLM).
"""

from .instruction_augmenter import InstructionAugmenter
from .llm_augmenter import LLMDirectInstructionAugmenter, build_llm_direct_augmenter
from .semantic_pathplanning_augmenter import (
    SemanticPathPlanningAugmenter,
    build_semantic_pathplanning_augmenter,
)
from .r_only_augmenter import ROnlyInstructionAugmenter, build_r_only_augmenter
from .template_augmenter import TemplatePathInstructionAugmenter, build_template_path_augmenter
from .types import (
    AugmentationResult,
    evidence_to_pretty_json,
    normalize_robot_caption,
    retrieval_evidence_from_plan,
    strip_code_fence,
)

__all__ = [
    "AugmentationResult",
    "InstructionAugmenter",
    "LLMDirectInstructionAugmenter",
    "SemanticPathPlanningAugmenter",
    "ROnlyInstructionAugmenter",
    "TemplatePathInstructionAugmenter",
    "build_llm_direct_augmenter",
    "build_semantic_pathplanning_augmenter",
    "build_r_only_augmenter",
    "build_template_path_augmenter",
    "evidence_to_pretty_json",
    "normalize_robot_caption",
    "retrieval_evidence_from_plan",
    "strip_code_fence",
]
