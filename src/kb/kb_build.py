# -*- coding: utf-8 -*-
"""
从外部数据构造 KB 目录。当前：``build_knowledgebase_from_memory``；
后续可扩展：从轨迹构建等。

每个 connectivity 节点（``image_id``）在 KB 中只对应 **一条** view：仅采用
``mp3d_view_annotation`` 里该节点的 **第一条** 标注（viewpoint）；不再为同一
``image_id`` 展开多条随机 view_id。``attach_view_images_to_kb`` 按 view 条目逐张渲染。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import secrets
import string
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from ..utils.io_utils import ensure_dir, list_json_in_dir, load_json, save_json

from .kb import (
    DOCUMENTS_FILENAME,
    MANIFEST_FILENAME,
    KnowledgeBase,
    SCENES_DIRNAME,
    SOURCE_MEMORY,
    scene_json_path,
)

# ---------------------------------------------------------------------------
# manifest 与空目录
# ---------------------------------------------------------------------------


def make_manifest(
    name: str,
    source: str,
    scene_ids: Optional[List[str]] = None,
    description: Optional[str] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "name": name,
        "source": source,
        "scenes": list(scene_ids or []),
        "description": description or "",
        "created_at": created_at or now,
        "updated_at": updated_at or now,
        **(extra or {}),
    }


def make_scene_description(
    scene_id: str,
    source: str,
    *,
    viewpoints: Optional[List[Dict[str, Any]]] = None,
    connectivity: Optional[List[Dict[str, Any]]] = None,
    views: Optional[Dict[str, Any]] = None,
    zones: Optional[Dict[str, Any]] = None,
    merged_from: Optional[List[str]] = None,
    trajectory_summary: Optional[List[Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"scene_id": scene_id, "source": source}
    if viewpoints is not None:
        out["viewpoints"] = viewpoints
    if connectivity is not None:
        out["connectivity"] = connectivity
    if views is not None:
        out["views"] = views
    if zones is not None:
        out["zones"] = zones
    if merged_from is not None:
        out["merged_from"] = merged_from
    if trajectory_summary is not None:
        out["trajectory_summary"] = trajectory_summary
    if extra:
        out["extra"] = extra
    return out


def save_manifest(
    kb_root: Union[str, Path],
    manifest: Dict[str, Any],
    *,
    update_timestamp: bool = True,
) -> None:
    if update_timestamp and "updated_at" in manifest:
        manifest = {**manifest, "updated_at": datetime.now(timezone.utc).isoformat()}
    path = Path(kb_root) / MANIFEST_FILENAME
    ensure_dir(path.parent)
    save_json(manifest, path)


def create_kb_folder(
    kb_folder: Union[str, Path],
    name: str,
    source: str,
    scene_ids: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    kb_folder = Path(kb_folder)
    ensure_dir(kb_folder)
    ensure_dir(kb_folder / SCENES_DIRNAME)
    manifest = make_manifest(
        name=name, source=source, scene_ids=scene_ids or [], description=description
    )
    save_manifest(kb_folder, manifest, update_timestamp=False)
    return manifest


# ---------------------------------------------------------------------------
# 构建入口
# ---------------------------------------------------------------------------


def build_knowledgebase_from_memory(
    memory_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    connectivity_subdirs: Optional[List[str]] = None,
    house_annotation_path: Optional[Union[str, Path]] = None,
    view_annotation_path: Optional[Union[str, Path]] = None,
    zone_annotation_path: Optional[Union[str, Path]] = None,
) -> KnowledgeBase:
    base = Path(memory_dir)
    if not base.exists():
        raise FileNotFoundError(f"Scene memory directory not found: {base}")
    out = Path(output_dir)
    ensure_dir(out)
    scenes_dir = ensure_dir(out / SCENES_DIRNAME)

    house_ann = _safe_load_dict(house_annotation_path)
    view_ann = _safe_load_dict(view_annotation_path)
    zone_ann = _safe_load_dict(zone_annotation_path)
    connectivity_subdirs = connectivity_subdirs or ["connectivity_mp3d"]

    json_paths: List[Path] = []
    for sub in connectivity_subdirs:
        sp = base / sub
        if sp.is_dir():
            json_paths.extend(list_json_in_dir(sp, "*.json"))

    scene_ids: List[str] = []
    documents: List[Dict[str, Any]] = []

    for jpath in tqdm(json_paths, desc="构建 KB 场景", unit="file"):
        scene_id = jpath.stem.replace("_connectivity", "").strip()
        try:
            connectivity = load_json(jpath)
        except Exception:
            continue
        if not isinstance(connectivity, list):
            continue
        tree = _build_scene_tree(
            scene_id=scene_id,
            connectivity=connectivity,
            house_entry=house_ann.get(scene_id) if house_ann else None,
            view_entry=view_ann.get(scene_id) if view_ann else None,
            zone_entry=zone_ann.get(scene_id) if zone_ann else None,
        )
        save_json(tree, scenes_dir / f"{scene_id}.json")
        scene_ids.append(scene_id)
        documents.extend(flatten_scene_to_docs(tree))

    manifest = make_manifest(
        name=out.name,
        source=SOURCE_MEMORY,
        scene_ids=scene_ids,
        description="KB: flat zones/views/instances + scene view_graph.",
    )
    save_manifest(out, manifest, update_timestamp=False)
    save_json(documents, out / DOCUMENTS_FILENAME)
    return KnowledgeBase(out)


def flatten_kb_documents(kb: KnowledgeBase) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sid in kb.scene_ids:
        try:
            out.extend(flatten_scene_to_docs(kb.scene(sid)))
        except Exception:
            continue
    return out


def attach_view_images_to_kb(
    kb: KnowledgeBase,
    scene_root: Union[str, Path],
    scene_ids: Optional[List[str]] = None,
    *,
    width: int = 640,
    height: int = 480,
    hfov: float = 90.0,
    verbose_skip_glb: bool = True,
) -> Tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("需要 PIL：pip install pillow") from e
    from ..utils.habitat_render import make_mp3d_sim, render_rgb_at_pose

    scene_root = Path(scene_root)
    ids = list(scene_ids) if scene_ids is not None else list(kb.scene_ids)
    saved = 0
    skipped = 0

    for scene_id in tqdm(ids, desc="渲染 view 图像", unit="scene"):
        spath = scene_json_path(kb.root, scene_id)
        if not spath.is_file():
            continue
        glb = scene_root / scene_id / f"{scene_id}.glb"
        if not glb.is_file():
            skipped += 1
            if verbose_skip_glb:
                print(f"[render] 跳过场景 {scene_id}：未找到 {glb}")
            continue

        data = load_json(spath)
        views = data.get("views")
        if not isinstance(views, dict) or not views:
            continue

        img_dir = kb.imgs_dir / scene_id
        img_dir.mkdir(parents=True, exist_ok=True)
        sim = make_mp3d_sim(glb, width, height, hfov)
        try:
            items = [(k, v) for k, v in views.items() if isinstance(v, dict)]
            for view_id, vnode in tqdm(
                items, desc=f"  {scene_id[:16]}", leave=False, unit="view"
            ):
                attrs = vnode.setdefault("attributes", {})
                pos, quat, hv = _pose_from_view_attrs(attrs)
                if pos is None or quat is None:
                    skipped += 1
                    continue
                if hv is not None:
                    pos = pos.copy()
                    pos[1] = float(hv)
                try:
                    rgb = render_rgb_at_pose(sim, pos, quat)
                except Exception as ex:  # pragma: no cover
                    print(f"\n[render] {scene_id}/{view_id}: {ex}")
                    skipped += 1
                    continue
                fname = f"{view_id}.png"
                Image.fromarray(rgb).save(img_dir / fname, format="PNG")
                attrs["img"] = f"imgs/{scene_id}/{fname}".replace("\\", "/")
                saved += 1
        finally:
            sim.close()
        save_json(data, spath)
        kb.invalidate_scene(scene_id)

    return saved, skipped


# --- 内部：树构建 ---


def _safe_load_dict(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    data = load_json(p)
    return data if isinstance(data, dict) else None


def _class_name_from_instance_key(key: str) -> str:
    if "_" in key:
        head, tail = key.rsplit("_", 1)
        if tail.isdigit():
            return head
    return key


def _instance_id(scene_id: str, view_id: str, inst_key: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in inst_key)
    return f"{view_id}_{safe}"


def _build_scene_tree(
    scene_id: str,
    connectivity: List[Dict[str, Any]],
    house_entry: Any,
    view_entry: Any,
    zone_entry: Any,
) -> Dict[str, Any]:
    if isinstance(house_entry, list) and house_entry and isinstance(house_entry[0], str):
        scene_description = house_entry[0]
    elif isinstance(house_entry, str):
        scene_description = house_entry
    else:
        scene_description = "" if house_entry is None else str(house_entry)

    zones, zone_adj, zone_ids_ordered, view_to_zone = _build_zones(zone_entry, scene_id)
    views = _build_views(scene_id, connectivity, view_entry, view_to_zone)
    _refresh_zone_view_ids(zones, views)
    view_ids_ordered = list(views.keys())
    scene_block = {
        "id": scene_id,
        "attributes": {
            "description": scene_description,
            "zone_ids": list(zone_ids_ordered),
            "zone_graph": {"adjacency": zone_adj},
            "view_ids": view_ids_ordered,
        },
    }
    return {
        "id": scene_id,
        "source": "memory",
        "scene": scene_block,
        "zones": zones,
        "views": views,
    }


def _build_zones(
    zone_entry: Any, scene_id: str
) -> Tuple[Dict[str, Any], List[List[int]], List[str], Dict[str, str]]:
    zones: Dict[str, Any] = {}
    zone_adj: List[List[int]] = []
    view_to_zone: Dict[str, str] = {}
    zone_ids_ordered: List[str] = []
    if (
        not isinstance(zone_entry, list)
        or len(zone_entry) != 2
        or not isinstance(zone_entry[0], list)
        or not isinstance(zone_entry[1], dict)
    ):
        return zones, zone_adj, zone_ids_ordered, view_to_zone

    adj_matrix, zone_dict = zone_entry
    zone_ids_ordered = list(zone_dict.keys())
    n = len(zone_ids_ordered)
    for i in range(n):
        row_src = adj_matrix[i] if i < len(adj_matrix) and isinstance(adj_matrix[i], list) else []
        row = [1 if (j < len(row_src) and row_src[j]) else 0 for j in range(n)]
        zone_adj.append(row)
    for i, zid in enumerate(zone_ids_ordered):
        neighbors = [
            zone_ids_ordered[j]
            for j in range(n)
            if j != i and j < len(zone_adj[i]) and zone_adj[i][j]
        ]
        value = zone_dict.get(zid)
        view_ids: List[str] = []
        text = ""
        if isinstance(value, list) and len(value) >= 2:
            cv, ct = value[0], value[1]
            if isinstance(cv, list):
                view_ids = [str(v) for v in cv]
            text = ct if isinstance(ct, str) else str(ct)
        for vid in view_ids:
            view_to_zone[vid] = zid
        zones[zid] = {
            "id": zid,
            "attributes": {
                "description": text,
                # 展开后将回填新的随机 view_id 列表
                "view_ids": [],
                "scene_id": scene_id,
                "adjacent_zone_ids": neighbors,
            },
        }
    return zones, zone_adj, zone_ids_ordered, view_to_zone


def _build_views(
    scene_id: str,
    connectivity: List[Dict[str, Any]],
    view_entry: Any,
    view_to_zone: Dict[str, str],
) -> Dict[str, Any]:
    views: Dict[str, Any] = {}
    used_ids: set = set()
    view_ids: List[str] = []
    connectivity_by_vid: Dict[str, Dict[str, Any]] = {}
    for node in connectivity:
        vid = str(node.get("image_id", ""))
        if vid:
            view_ids.append(vid)
            connectivity_by_vid[vid] = node
    scene_view_ann: Dict[str, Any] = view_entry if isinstance(view_entry, dict) else {}
    for base_vid in view_ids:
        if not base_vid:
            continue
        node = connectivity_by_vid.get(base_vid, {})
        ann_list = scene_view_ann.get(base_vid, [])
        ann_dicts = [a for a in ann_list if isinstance(a, dict)] if isinstance(ann_list, list) else []
        cand: Dict[str, Any] = ann_dicts[0] if ann_dicts else {}
        desc = cand.get("view_summary", "") or ""
        ann_pos, ann_rot = cand.get("habitat_position"), cand.get("habitat_rotation")
        position = rotation = None
        if (
            isinstance(ann_pos, list)
            and len(ann_pos) == 3
            and isinstance(ann_rot, list)
            and len(ann_rot) == 4
        ):
            position = [float(ann_pos[0]), float(ann_pos[1]), float(ann_pos[2])]
            rotation = [float(ann_rot[0]), float(ann_rot[1]), float(ann_rot[2]), float(ann_rot[3])]
        else:
            pose16 = node.get("pose", [])
            if isinstance(pose16, list) and len(pose16) == 16:
                position, rotation = _pose16_to_position_rotation(pose16)
        vid = _new_random_view_id(used_ids)
        zid = view_to_zone.get(base_vid)
        attrs: Dict[str, Any] = {
            "description": str(desc),
            "position": position,
            "rotation": rotation,
            "included": node.get("included", True),
            "zone_id": zid,
            "img": None,
            "base_view_id": base_vid,
        }
        if "height" in node:
            attrs["height"] = node["height"]
        views[vid] = {"id": vid, "attributes": attrs}
    return views


def _new_random_view_id(used: set) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        v = "".join(secrets.choice(alphabet) for _ in range(32))
        if v not in used:
            used.add(v)
            return v


def _refresh_zone_view_ids(zones: Dict[str, Any], views: Dict[str, Any]) -> None:
    zone_to_views: Dict[str, List[str]] = {}
    for vid, vnode in views.items():
        attrs = vnode.get("attributes") if isinstance(vnode, dict) else {}
        if not isinstance(attrs, dict):
            continue
        zid = attrs.get("zone_id")
        if isinstance(zid, str) and zid:
            zone_to_views.setdefault(zid, []).append(str(vid))
    for zid, znode in zones.items():
        if not isinstance(znode, dict):
            continue
        zattrs = znode.get("attributes")
        if isinstance(zattrs, dict):
            zattrs["view_ids"] = zone_to_views.get(str(zid), [])


def _pose16_to_position_rotation(pose16: List[float]) -> Tuple[List[float], List[float]]:
    r00, r01, r02, tx = pose16[0], pose16[1], pose16[2], pose16[3]
    r10, r11, r12, ty = pose16[4], pose16[5], pose16[6], pose16[7]
    r20, r21, r22, tz = pose16[8], pose16[9], pose16[10], pose16[11]
    t = r00 + r11 + r22
    if t > 0.0:
        S = (t + 1.0) ** 0.5 * 2.0
        qw, qx, qy, qz = 0.25 * S, (r21 - r12) / S, (r02 - r20) / S, (r10 - r01) / S
    elif (r00 > r11) and (r00 > r22):
        S = (1.0 + r00 - r11 - r22) ** 0.5 * 2.0
        qw, qx, qy, qz = (r21 - r12) / S, 0.25 * S, (r01 + r10) / S, (r02 + r20) / S
    elif r11 > r22:
        S = (1.0 + r11 - r00 - r22) ** 0.5 * 2.0
        qw, qx, qy, qz = (r02 - r20) / S, (r01 + r10) / S, 0.25 * S, (r12 + r21) / S
    else:
        S = (1.0 + r22 - r00 - r11) ** 0.5 * 2.0
        qw, qx, qy, qz = (r10 - r01) / S, (r02 + r20) / S, (r12 + r21) / S, 0.25 * S
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm > 1e-8:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return [float(tx), float(ty), float(tz)], [float(qx), float(qy), float(qz), float(qw)]


def flatten_scene_to_docs(scene_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    scene_id = scene_tree.get("id", "")
    scene_desc = (scene_tree.get("scene") or {}).get("attributes", {}).get("description", "")
    zones = scene_tree.get("zones") if isinstance(scene_tree.get("zones"), dict) else {}
    views = scene_tree.get("views") if isinstance(scene_tree.get("views"), dict) else {}
    instances = scene_tree.get("instances") if isinstance(scene_tree.get("instances"), dict) else {}
    zone_desc_map = {
        zid: (zv.get("attributes", {}).get("description", "") or "")
        for zid, zv in zones.items()
    }
    view_to_inst: Dict[str, List[str]] = {}
    for inode in instances.values():
        a = inode.get("attributes") or {}
        vid = a.get("view_id")
        if not vid:
            continue
        cn, d = a.get("class_name", ""), a.get("description", "")
        line = f"{cn}: {d}".strip() if cn else str(d)
        view_to_inst.setdefault(str(vid), []).append(line)
    for vid, vnode in views.items():
        attrs = vnode.get("attributes", {})
        zid = attrs.get("zone_id")
        zdesc = zone_desc_map.get(zid, "") if zid else ""
        vdesc = attrs.get("description", "") or ""
        inst_blob = " ".join(view_to_inst.get(vid, []))
        text = " ".join(p for p in [scene_desc, zdesc, vdesc, inst_blob] if p)
        docs.append(
            {
                "scene_id": scene_id,
                "node_type": "view",
                "view_id": vid,
                "zone_id": zid,
                "text": text,
            }
        )
    return docs


def _pose_from_view_attrs(attrs: Dict[str, Any]):
    import numpy as np

    pos, rot, height_val = attrs.get("position"), attrs.get("rotation"), attrs.get("height")
    if not (isinstance(pos, list) and len(pos) == 3):
        return None, None, None
    position = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
    # 扁平化后每个新 view_id 理应只有一个 rotation；为兼容旧数据，列表嵌套时取第一个
    if isinstance(rot, list) and rot and isinstance(rot[0], list):
        rot = rot[0]
    if not (isinstance(rot, list) and len(rot) == 4):
        return None, None, None
    quat = np.array([float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])], dtype=np.float32)
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    hv = float(height_val) if isinstance(height_val, (int, float)) else None
    return position, quat, hv
