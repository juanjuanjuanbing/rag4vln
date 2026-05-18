# -*- coding: utf-8 -*-
"""
不完全指令增强：检索证据 + 简短意图 → 可执行导航指令。

- ``InstructionAugmenter``：基类。
- ``LLMDirectInstructionAugmenter``：第一种实现，单次大模型生成。
- ``TemplatePathInstructionAugmenter``：第二种，LLM 填槽 + 本地模板拼接。
- ``SemanticPathPlanningAugmenter``：第三种，语义路径规划（三阶段 CoT + 最终 VLN 指令）。
- ``ROnlyInstructionAugmenter``：R_only baseline，仅拼接检索证据与原始指令（无 LLM）。
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
