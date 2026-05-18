# -*- coding: utf-8 -*-
"""
知识库实体：指向标准 KB 目录；``kb.scene(scene_id)`` 返回该场景完整 JSON（dict）。
zone / view 等从返回的字典里取，例如 ``tree["zones"][zid]``、``tree["views"][vid]``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.io_utils import load_json

MANIFEST_FILENAME = "manifest.json"
SCENES_DIRNAME = "scenes"
SCENE_FILE_SUFFIX = ".json"
DOCUMENTS_FILENAME = "documents.json"
SOURCE_MEMORY = "memory"
SOURCE_EXPERIENCE = "experience"


def scene_json_path(kb_root: Union[str, Path], scene_id: str) -> Path:
    # MP3D 场景 id 里可能出现括号后缀（如 "XcA2TqTSSAj(1)"），需允许。
    safe = "".join(c for c in scene_id if c.isalnum() or c in "-_()")
    if safe != scene_id:
        raise ValueError(f"Invalid scene_id: {scene_id!r}")
    return Path(kb_root) / SCENES_DIRNAME / f"{scene_id}{SCENE_FILE_SUFFIX}"


class KnowledgeBase:
    """
    标准 KB 目录：``manifest.json``、``scenes/``、``documents.json``、可选 ``imgs/``。

    - ``scene_ids``：初始化时从 manifest 读取。
    - ``scene(id)``：返回与 ``scenes/<id>.json`` 一致的字典（含 ``scene`` / ``zones`` / ``views`` / ``instances``）。
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self._scene_cache: Dict[str, Dict[str, Any]] = {}
        mp = self.root / MANIFEST_FILENAME
        if mp.is_file():
            self._manifest: Dict[str, Any] = load_json(mp)
        else:
            self._manifest = {}
        self.scene_ids: List[str] = list(self._manifest.get("scenes", []))

    def manifest(self) -> Dict[str, Any]:
        return dict(self._manifest)

    def invalidate_scene(self, scene_id: Optional[str] = None) -> None:
        """丢弃缓存的场景 JSON（例如磁盘上刚更新过）。"""
        if scene_id is None:
            self._scene_cache.clear()
        else:
            self._scene_cache.pop(scene_id, None)

    def scene(self, scene_id: str) -> Dict[str, Any]:
        """加载并返回场景树（dict），与磁盘 JSON 结构一致。"""
        if scene_id not in self._scene_cache:
            path = scene_json_path(self.root, scene_id)
            if not path.is_file():
                raise FileNotFoundError(f"Scene JSON not found: {path}")
            self._scene_cache[scene_id] = load_json(path)
        return self._scene_cache[scene_id]

    @property
    def scenes_dir(self) -> Path:
        return self.root / SCENES_DIRNAME

    @property
    def imgs_dir(self) -> Path:
        return self.root / "imgs"

    def view_image_path(self, scene_id: str, view_id: str) -> Optional[Path]:
        """某 view 的 ``attributes.img`` 相对 KB 根解析后的绝对路径；img 为列表时取第一张。"""
        views = self.scene(scene_id).get("views")
        if not isinstance(views, dict):
            return None
        vnode = views.get(view_id)
        if not isinstance(vnode, dict):
            return None
        rel = (vnode.get("attributes") or {}).get("img")
        if isinstance(rel, list):
            rel = rel[0] if rel else None
        if not rel or not isinstance(rel, str):
            return None
        return (self.root / rel.replace("\\", "/")).resolve()

    def load_view_image(self, scene_id: str, view_id: str):
        """若存在渲染图且可读，返回 PIL RGB Image，否则 None。"""
        p = self.view_image_path(scene_id, view_id)
        if p is None or not p.is_file():
            return None
        try:
            from PIL import Image

            return Image.open(p).convert("RGB")
        except Exception:
            return None

    def list_scene_ids(self) -> List[str]:
        return list(self.scene_ids)

    def retrieval_documents(self) -> List[Dict[str, Any]]:
        from .kb_build import flatten_kb_documents

        doc_path = self.root / DOCUMENTS_FILENAME
        if doc_path.is_file():
            data = load_json(doc_path)
            return data if isinstance(data, list) else []
        return flatten_kb_documents(self)

    def get_documents(self) -> List[Dict[str, Any]]:
        return self.retrieval_documents()

    def num_documents(self) -> int:
        return len(self.retrieval_documents())
