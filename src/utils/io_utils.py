# -*- coding: utf-8 -*-
"""
工具函数：数据读写等。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json


def load_json(path: Union[str, Path]) -> Any:
    """加载 JSON 文件。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: Optional[int] = 2) -> None:
    """保存为 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def list_json_in_dir(
    dir_path: Union[str, Path],
    pattern: str = "*.json",
    recursive: bool = False,
) -> List[Path]:
    """列出目录下（可选递归）的 JSON 文件。"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    if recursive:
        return sorted(dir_path.rglob(pattern))
    return sorted(dir_path.glob(pattern))


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，不存在则创建。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
