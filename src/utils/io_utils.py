# -*- coding: utf-8 -*-
"""
Utility functions: data I/O, etc.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: Optional[int] = 2) -> None:
    """Save data as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def list_json_in_dir(
    dir_path: Union[str, Path],
    pattern: str = "*.json",
    recursive: bool = False,
) -> List[Path]:
    """List JSON files under a directory (optionally recursive)."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    if recursive:
        return sorted(dir_path.rglob(pattern))
    return sorted(dir_path.glob(pattern))


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists; create if missing."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
