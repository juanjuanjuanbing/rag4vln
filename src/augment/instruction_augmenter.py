# -*- coding: utf-8 -*-
"""
不完全导航指令增强：抽象基类。

输入为简短用户意图（如「去床边」）以及检索器给出的结构化证据（场景 / 区域 / 起终点视角、路径、分数、可选机器人观测描述）。
输出为 ``AugmentationResult``，其中 ``instruction`` 为扩写后的可执行指令。

后续可在此包内增加第二种、第三种具体增强器（模板融合、检索片段拼装等），均继承本基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional

from .types import AugmentationResult, normalize_robot_caption


class InstructionAugmenter(ABC):
    """
    指令增强器基类。

    ``evidence`` 建议使用 ``retrieval_evidence_from_plan(retriever_output)`` 生成，
    或与之同结构的 dict（含 ``topk1_scenes`` / ``topk2_zones`` / ``topk3_pairs`` / ``robot_caption``）。
    """

    @abstractmethod
    def augment(
        self,
        instruction: str,
        evidence: Mapping[str, Any],
        *,
        robot_caption: Optional[str] = None,
        path_region_descriptions: Optional[list[str]] = None,
    ) -> AugmentationResult:
        """
        :param instruction: 信息不充分的用户意图
        :param evidence: 检索证据（含置信度与路径）
        :param robot_caption: 机器人观测文本描述；若为 None 则从 ``evidence['robot_caption']`` 读取并规范化
        :param path_region_descriptions: 选中路径经过的区域（如 zone）的人类可读描述；通常需要通过 KB 由 view path 映射得到
        """
        raise NotImplementedError

    @staticmethod
    def default_config_path() -> Path:
        from ..config_io import DEFAULT_CONFIG_PATH

        return DEFAULT_CONFIG_PATH

    @staticmethod
    def resolve_robot_caption(
        evidence: Mapping[str, Any],
        robot_caption: Optional[str],
    ) -> str:
        cap = robot_caption
        if cap is None:
            cap = evidence.get("robot_caption")
        return normalize_robot_caption(cap if cap is not None else "")
