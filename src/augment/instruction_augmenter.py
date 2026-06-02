# -*- coding: utf-8 -*-
"""
Incomplete navigation instruction augmentation: abstract base class.

Input: short user intent (e.g. "go to the bed") plus structured retrieval evidence
(scenes / zones / start-end views, path, scores, optional robot observation text).
Output: ``AugmentationResult`` with an expanded executable ``instruction``.

Additional augmenters (template fusion, evidence stitching, etc.) subclass this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional

from .types import AugmentationResult, normalize_robot_caption


class InstructionAugmenter(ABC):
    """
    Base class for instruction augmenters.

    ``evidence`` should come from ``retrieval_evidence_from_plan(retriever_output)``
    or a dict with the same keys (``topk1_scenes`` / ``topk2_zones`` / ``topk3_pairs`` / ``robot_caption``).
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
        :param instruction: underspecified user intent
        :param evidence: retrieval evidence (scores and path)
        :param robot_caption: robot observation text; if None, read from ``evidence['robot_caption']`` and normalize
        :param path_region_descriptions: human-readable regions along the selected path (often mapped from KB view path)
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
