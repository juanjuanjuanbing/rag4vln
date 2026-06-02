# -*- coding: utf-8 -*-
"""Instruction augmentation: result types and retrieval evidence normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class AugmentationResult:
    """Augmented instruction and metadata."""

    instruction: str
    """Expanded natural-language instruction suitable for navigation."""
    raw_model_output: Optional[str] = None
    """Raw model output when available."""
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_robot_caption(text: Optional[str]) -> str:
    """
    Normalize `robot_caption` to plain text (handles VLMs that wrap descriptions in JSON or odd structures).
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and len(obj) == 1:
            k, v = next(iter(obj.items()))
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(k, str) and len(k) > 12:
                return k.strip()
        if isinstance(obj, str):
            return obj.strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Double-quoted form like "{\"...\"}"
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        inner = s[1:-1].replace('\\"', '"').strip()
        if inner:
            return normalize_robot_caption(inner)
    return s


def retrieval_evidence_from_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract structured evidence from ``Retriever.retrieve`` for augmentation (drops timing, etc.).

    Kept fields:
    - ``robot_caption``
    - ``topk1_scenes``: scene ids and ``scene_belonging_score``
    - ``topk2_zones``: start-zone candidates and ``start_zone_belonging_score``
    - ``topk3_pairs``: start/end views, ``scores``, ``path``, etc.
    """
    keys = (
        "robot_caption",
        "topk1_scenes",
        "topk2_zones",
        "topk3_pairs",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if k in plan:
            out[k] = plan[k]
    return out


def evidence_to_pretty_json(evidence: Mapping[str, Any]) -> str:
    """JSON string for prompts (excludes robot_caption, shown separately)."""
    payload = {k: v for k, v in evidence.items() if k != "robot_caption"}
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def strip_code_fence(text: str) -> str:
    """Strip markdown code fences if the model wrapped its output."""
    s = text.strip()
    m = re.match(r"^```(?:\w+)?\s*\n?(.*)\n?```\s*$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s
