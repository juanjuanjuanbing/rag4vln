# -*- coding: utf-8 -*-
"""
Augmenter type 3: semantic path planning (single LLM, three-stage CoT).
Prefer parsing ``<final_instruction>...</final_instruction>``; if tags are missing or output
is truncated, fall back with heuristics so the eval pipeline does not abort.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

from .instruction_augmenter import InstructionAugmenter
from .types import AugmentationResult, evidence_to_pretty_json, strip_code_fence


def _resolve_api_key(key_spec: str) -> str:
    """Resolve API key from environment variable name only (see config ``api_key_env``)."""
    if not key_spec:
        return ""
    name = key_spec.strip()
    return os.environ.get(name, "")


def _load_merged_semantic_pathplanning_cfg(config_path: Optional[Union[str, Path]]) -> dict:
    from ..config_io import load_augment_config

    p = Path(config_path) if config_path is not None else InstructionAugmenter.default_config_path()
    aug = load_augment_config(p) if p.is_file() else {}
    llm = aug.get("llm") or {}
    sp = dict(aug.get("semantic_pathplanning") or {})
    for k in ("enabled", "api_key_env", "base_url", "model", "timeout_sec", "temperature"):
        if k not in sp or sp[k] in (None, ""):
            if k in llm and llm[k] not in (None, ""):
                sp[k] = llm[k]
    return sp


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_final_instruction(raw: str) -> tuple[str, str]:
    """
    Extract the final navigation instruction from the model reply.

    Returns:
        (instruction, source)  source for meta debugging: tag | unclosed_tag | heading | tail
    """
    t = (raw or "").strip()
    if not t:
        return "", "empty"

    inst = _extract_tag(t, "final_instruction")
    if inst:
        return inst, "tag"

    # Open tag only, no close tag (common when max_tokens truncates)
    m_open = re.search(r"<final_instruction>\s*(.*)", t, re.DOTALL | re.IGNORECASE)
    if m_open:
        s = m_open.group(1).strip()
        if "</final_instruction>" in s:
            s = s.split("</final_instruction>", 1)[0].strip()
        if s:
            return s, "unclosed_tag"

    # Common "heading + body" variants when the model skips XML
    for pat in (
        r"(?:^|\n)\s*#+\s*final\s*instruction\s*(?:\n+)(.+?)(?=\n\s*#+|\n\s*<|\Z)",
        r"(?:^|\n)\s*\*\*final\s*instruction\*\*\s*(?:\n+)(.+?)(?=\n\s*#+|\n\s*\*\*|\n\s*<|\Z)",
        r"(?:^|\n)\s*final\s*instruction\s*[:：]\s*(.+?)(?=\n\n|\Z)",
    ):
        m = re.search(pat, t, re.DOTALL | re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            # Use first contiguous non-empty lines as one instruction
            first_para = "\n".join(line.strip() for line in cand.splitlines() if line.strip())
            if first_para and len(first_para) >= 8:
                return first_para.split("\n", 1)[0].strip(), "heading"

    # Weak fallback: last non-empty paragraph (often the final instruction)
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    if paras:
        last = paras[-1]
        # Strip leftover XML tag fragments
        last = re.sub(r"</?[^>]+>", "", last).strip()
        if last and len(last) >= 8 and not last.lower().startswith("put reasoning"):
            return last, "tail"

    return "", "none"


def _parse_waypoints_block(block: str) -> Optional[list]:
    """
    Parse JSON inside ``<waypoints_json>``. Supports:
    - Canonical: ``{"waypoints":[...]}``
    - Drift: top-level ``[{...}, ...]`` array
    - Alternate keys: ``waypoint`` / ``path`` / ``semantic_waypoints`` / ``steps``
    Returns ``None`` on failure (does not abort augmentation).
    """
    s = strip_code_fence(block.strip())
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None

    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            return obj
        return None

    if not isinstance(obj, dict):
        return None

    w = obj.get("waypoints")
    if isinstance(w, list):
        return w

    for k in ("waypoint", "path", "semantic_waypoints", "steps", "waypoints_list"):
        v = obj.get(k)
        if isinstance(v, list):
            return v

    # Single waypoint object (rare)
    if "kind" in obj or "semantic" in obj:
        return [obj]

    return None


class SemanticPathPlanningAugmenter(InstructionAugmenter):
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self._cfg = _load_merged_semantic_pathplanning_cfg(config_path)
        self._config_path = Path(config_path) if config_path is not None else self.default_config_path()
        self._client: Optional[Any] = None
        self._client_inited = False

    def _lazy_init_client(self) -> None:
        if self._client_inited:
            return
        self._client_inited = True
        if not self._cfg.get("enabled", True):
            raise RuntimeError("augment.semantic_pathplanning.enabled is false")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            raise RuntimeError("openai package is not installed for semantic_pathplanning") from e
        base = str(self._cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        key = _resolve_api_key(str(self._cfg.get("api_key_env", "DASHSCOPE_API_KEY")))
        if not key:
            raise RuntimeError("missing semantic_pathplanning api key")
        self._client = OpenAI(api_key=key, base_url=base)

    def augment(
        self,
        instruction: str,
        evidence: Mapping[str, Any],
        *,
        robot_caption: Optional[str] = None,
        path_region_descriptions: Optional[List[str]] = None,
    ) -> AugmentationResult:
        cap = self.resolve_robot_caption(evidence, robot_caption)
        evidence_json = evidence_to_pretty_json(evidence)
        regions = path_region_descriptions or []
        path_region_descriptions_json = json.dumps(
            [str(x) for x in regions if x is not None and str(x).strip() != ""],
            ensure_ascii=False,
        )
        pair0: Mapping[str, Any] = {}
        pairs = evidence.get("topk3_pairs")
        if isinstance(pairs, list) and pairs and isinstance(pairs[0], dict):
            pair0 = pairs[0]
        pz = pair0.get("path_zone_ids")
        path_zone_ids_json = json.dumps(
            [str(x) for x in pz] if isinstance(pz, list) else [],
            ensure_ascii=False,
        )

        sys_p = str(self._cfg.get("system_prompt", "")).strip()
        user_tmpl = str(self._cfg.get("user_prompt_template", "")).strip()
        model = str(self._cfg.get("model", "qwen-plus"))
        if not sys_p or not user_tmpl:
            raise RuntimeError("missing prompts in semantic_pathplanning config")

        self._lazy_init_client()
        if self._client is None:
            raise RuntimeError("semantic_pathplanning client is not initialized")

        user_content = user_tmpl.format(
            instruction=(instruction or "").strip(),
            robot_caption=cap if cap else "None",
            evidence_json=evidence_json,
            path_region_descriptions_json=path_region_descriptions_json,
            path_zone_ids_json=path_zone_ids_json,
        )

        timeout = float(self._cfg.get("timeout_sec", 120))
        max_tokens = int(self._cfg.get("max_tokens", 2048))
        temperature = float(self._cfg.get("temperature", 0.3))
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            raise RuntimeError("semantic_pathplanning empty content")

        final_inst, final_src = _extract_final_instruction(raw)
        if not final_inst:
            raise RuntimeError(
                "semantic_pathplanning missing <final_instruction> tag and no usable fallback; "
                "check model output / max_tokens / prompt compliance."
            )

        task_narrative = _extract_tag(raw, "task_narrative")
        wblock = _extract_tag(raw, "waypoints_json")
        waypoints = _parse_waypoints_block(wblock) if wblock else None

        meta: dict[str, Any] = {
            "augmenter": "semantic_pathplanning",
            "model": model,
            "config": str(self._config_path),
            "final_instruction_source": final_src,
        }
        if task_narrative:
            meta["task_narrative"] = task_narrative
        if waypoints is not None:
            meta["waypoints"] = waypoints

        return AugmentationResult(
            instruction=final_inst.strip(),
            raw_model_output=raw,
            meta=meta,
        )


def build_semantic_pathplanning_augmenter(
    config_path: Optional[Union[str, Path]] = None,
) -> SemanticPathPlanningAugmenter:
    return SemanticPathPlanningAugmenter(config_path=config_path)
