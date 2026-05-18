# -*- coding: utf-8 -*-
"""指令增强：结果类型与检索证据规范化。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class AugmentationResult:
    """增强后的指令及元信息。"""

    instruction: str
    """扩写后的、宜于导航执行的自然语言指令。"""
    raw_model_output: Optional[str] = None
    """模型原始输出（若有）。"""
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_robot_caption(text: Optional[str]) -> str:
    """
    将 `robot_caption` 规整为纯文本（兼容部分 VLM 把整段描述塞进 JSON 或异常结构的情况）。
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
    # 形如 "{\"...\"}" 的双层引号
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        inner = s[1:-1].replace('\\"', '"').strip()
        if inner:
            return normalize_robot_caption(inner)
    return s


def retrieval_evidence_from_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """
    从 ``Retriever.retrieve`` 的返回 dict 中抽取用于增强的结构化证据（去掉 timing 等）。

    保留字段：
    - ``robot_caption``
    - ``topk1_scenes``：场景 id 与 ``scene_belonging_score``
    - ``topk2_zones``：起始区域候选与 ``start_zone_belonging_score``
    - ``topk3_pairs``：起终点 view、``scores``、``path`` 等
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
    """供 prompt 使用的 JSON 字符串（排除 robot_caption，单独展示）。"""
    payload = {k: v for k, v in evidence.items() if k != "robot_caption"}
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def strip_code_fence(text: str) -> str:
    """若模型用 markdown 代码块包裹输出，去掉围栏。"""
    s = text.strip()
    m = re.match(r"^```(?:\w+)?\s*\n?(.*)\n?```\s*$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s
