# -*- coding: utf-8 -*-
"""
统一 YAML：`config.yaml` 内分 `basic` / `retrieval` / `augment`。
兼容旧版「仅检索」或「仅增强」的扁平单文件（顶层直接是各段字段）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError("config_io requires pyyaml") from e

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"


def load_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def normalize_retrieval_section(root: Dict[str, Any]) -> Dict[str, Any]:
    if not root:
        return {}
    r = root.get("retrieval")
    if isinstance(r, dict) and r:
        return dict(r)
    skip = {"basic", "augment", "retrieval"}
    if any(k in root for k in ("text_embedder", "embedding_dim", "caption", "vit", "bert")):
        return {k: v for k, v in root.items() if k not in skip}
    return {}


def normalize_augment_section(root: Dict[str, Any]) -> Dict[str, Any]:
    if not root:
        return {}
    a = root.get("augment")
    if isinstance(a, dict) and a:
        return dict(a)
    keys = ("llm", "template_llm", "semantic_pathplanning")
    if any(k in root for k in keys):
        return {k: root[k] for k in keys if k in root}
    return {}


def load_retrieval_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return normalize_retrieval_section(load_yaml_file(p))


def load_augment_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return normalize_augment_section(load_yaml_file(p))


def retrieval_embedding_dim(path: Optional[Union[str, Path]] = None) -> int:
    r = load_retrieval_config(path)
    return int(r.get("embedding_dim", 768))
