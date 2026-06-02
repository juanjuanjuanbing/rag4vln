#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build GT alignment for VLN-CE episodes (gt_scene / gt_start_view / gt_end_view).

**How gt_end_view_id is defined**
  1. Collect all usable views in the KB scene (with position, included=True).
  2. ``_pick_end_ref(episode)`` yields a 3D reference: prefer the **last** point of ``reference_path``,
     else ``goals[0].position``; if neither exists, fall back to ``start_position``.
  3. ``_nearest_view(views, end_ref)``: Euclidean-nearest KB view to that reference is GT end view.

Default (eval): scan only ``raw_data`` under vln-root, val_seen / val_unseen only (exclude train);
duplicate episode_id across JSON files in raw_data → keep one row per episode_id.
mask dirs (raw_data_mask_*) share episode ids with raw_data; not scanned by default to avoid dupes.

Default output:
  data/vln_ce/dataset_gt.csv
With rendering on (default), start and end PNGs are written. Default ``--start-view-subdir-style from_r2r``:
``raw_data`` / ``raw_data_mask_*`` / ``raw_data_implicit`` **share** images at
``data/vln_ce/start_view/r2r/<split>/ep_<id>.png``, matching eval script layout.
For implicit data, set scan dir e.g. ``--vln-subdir raw_data_implicit`` (default scans ``raw_data`` only).
``--no-render-end-view`` for start only; ``--no-render-start-view`` to skip all rendering.

Run from InternNav repo root, e.g.:
  python rag4vln/scripts/build_dataset_gt.py
  python rag4vln/scripts/build_dataset_gt.py --vln-subdir raw_data_implicit   # implicit instructions
  python rag4vln/scripts/build_dataset_gt.py --no-render-end-view   # start views only
  python rag4vln/scripts/build_dataset_gt.py --start-view-subdir-style mirror_vln_ce
  python rag4vln/scripts/build_dataset_gt.py --start-view-subdir-style mirror_json
  python rag4vln/scripts/build_dataset_gt.py --full-vln-ce
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

KB_AVG_CAMERA_HEIGHT = 1.5047912069468614


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_sys_path(repo_root: Path) -> None:
    rag4vln_root = repo_root / "rag4vln"
    if str(rag4vln_root) not in sys.path:
        sys.path.insert(0, str(rag4vln_root))


def _load_json(path: Path) -> Dict[str, Any]:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_vln_files(vln_root: Path) -> Iterable[Path]:
    files = sorted(vln_root.rglob("*.json"))
    gz_files = sorted(p for p in vln_root.rglob("*.gz") if p.name.endswith(".json.gz"))
    for p in files:
        yield p
    for p in gz_files:
        yield p


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _file_matches_split_filter(
    file_path: Path, scan_root: Path, split_dir_names: List[str], exclude_dir_names: List[str]
) -> bool:
    """Path under scan_root must hit at least one split dir name and none of the exclude names."""
    try:
        rel = file_path.relative_to(scan_root)
    except ValueError:
        return False
    parts = set(rel.parts[:-1])
    if any(x in parts for x in exclude_dir_names):
        return False
    return any(x in parts for x in split_dir_names)


def _split_tag_from_rel(rel: Path, split_dir_names: List[str]) -> str:
    """Dedup tag: same episode_id may appear in val_seen and val_unseen (different instruction/start); dedupe per split."""
    for name in split_dir_names:
        if name in rel.parts:
            return name
    return ""


def _parse_kb_scene_id(vln_scene_id: Any) -> str:
    if not isinstance(vln_scene_id, str):
        return ""
    s = vln_scene_id.strip().replace("\\", "/")
    if not s:
        return ""
    parts = [x for x in s.split("/") if x]
    if not parts:
        return ""
    last = parts[-1]
    if last.endswith(".glb"):
        return last[:-4]
    if len(parts) >= 2:
        return parts[-1]
    return last


def _parse_scene_file_scene_id(vln_scene_id: Any) -> str:
    if not isinstance(vln_scene_id, str):
        return ""
    s = vln_scene_id.strip().replace("\\", "/")
    if not s:
        return ""
    parts = [x for x in s.split("/") if x]
    if not parts:
        return ""
    if len(parts) >= 2 and parts[-1].endswith(".glb"):
        return parts[-2]
    return Path(parts[-1]).stem


def _to_xyz(vec: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(vec, (list, tuple)) or len(vec) < 3:
        return None
    try:
        return (float(vec[0]), float(vec[1]), float(vec[2]))
    except Exception:
        return None


def _to_quat_xyzw(vec: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(vec, (list, tuple)) or len(vec) < 4:
        return None
    try:
        return (float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3]))
    except Exception:
        return None


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _extract_views(scene_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    views = scene_tree.get("views") or {}
    if not isinstance(views, dict):
        return out
    for vid, node in views.items():
        attrs = (node or {}).get("attributes") or {}
        pos = _to_xyz(attrs.get("position"))
        if pos is None:
            continue
        if attrs.get("included") is False:
            continue
        out.append(
            {
                "view_id": str(vid),
                "zone_id": str(attrs.get("zone_id") or ""),
                "position": pos,
            }
        )
    return out


def _nearest_view(
    candidate_views: List[Dict[str, Any]], target: Optional[Tuple[float, float, float]]
) -> Tuple[str, str, Optional[Tuple[float, float, float]], Optional[float]]:
    if target is None or not candidate_views:
        return "", "", None, None
    best = None
    best_d = None
    for v in candidate_views:
        d = _dist(v["position"], target)
        if best_d is None or d < best_d:
            best = v
            best_d = d
    if best is None:
        return "", "", None, None
    return str(best["view_id"]), str(best["zone_id"]), tuple(best["position"]), float(best_d)


def _pick_end_ref(episode: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    ref_path = episode.get("reference_path")
    if isinstance(ref_path, list) and ref_path:
        p = _to_xyz(ref_path[-1])
        if p is not None:
            return p
    goals = episode.get("goals")
    if isinstance(goals, list) and goals:
        p = _to_xyz((goals[0] or {}).get("position"))
        if p is not None:
            return p
    return None


def _image_subdir_under_vln_view_root(rel_file: str, style: str, repo_root: Path) -> Path:
    """
    Relative subdir under start_view / end_view roots.
    - from_r2r (default): from ``r2r`` to JSON parent; all three dataset variants share ``start_view/r2r/<split>/``.
    - mirror_vln_ce: JSON parent path relative to ``data/vln_ce``.
    - mirror_json: full ``Path(rel_file).parent`` relative to repo (legacy nested data/vln_ce/...).
    """
    if style == "mirror_json":
        return Path(rel_file).parent
    if style == "mirror_vln_ce":
        vln_ce = (repo_root / "data" / "vln_ce").resolve()
        parent = (repo_root / rel_file).resolve().parent
        try:
            return parent.relative_to(vln_ce)
        except ValueError:
            return Path(rel_file).parent
    parts = list(Path(rel_file).parts)
    try:
        i = parts.index("r2r")
        return Path(*parts[i:-1])
    except ValueError:
        return Path(rel_file).parent


def _quat_xyzw_yaw_about_y(yaw_rad: float) -> Tuple[float, float, float, float]:
    """Rotation about Y (radians), Habitat-style x,y,z,w quaternion."""
    half = 0.5 * yaw_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def _end_rotation_xyzw_from_episode(ep: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """End camera orientation: forward on xz from last reference_path segment; else start_rotation."""
    ref_path = ep.get("reference_path")
    if isinstance(ref_path, list) and len(ref_path) >= 2:
        a = _to_xyz(ref_path[-2])
        b = _to_xyz(ref_path[-1])
        if a is not None and b is not None:
            dx = float(b[0] - a[0])
            dz = float(b[2] - a[2])
            if abs(dx) + abs(dz) > 1e-6:
                yaw = math.atan2(dx, dz)
                return _quat_xyzw_yaw_about_y(yaw)
    q = _to_quat_xyzw(ep.get("start_rotation"))
    if q is not None:
        return q
    return None


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Build GT mapping CSV for VLN-CE dataset")
    parser.add_argument("--vln-root", type=Path, default=repo_root / "data" / "vln_ce")
    parser.add_argument(
        "--full-vln-ce",
        action="store_true",
        help="Scan entire vln-root (incl. raw_data_mask_*); no split filter or episode dedup",
    )
    parser.add_argument(
        "--vln-subdir",
        type=str,
        default="raw_data",
        help="Scan only under this subdir of vln-root (default); --full-vln-ce overrides",
    )
    parser.add_argument(
        "--split-dirs",
        type=str,
        default="val_seen,val_unseen",
        help="Only JSON whose path contains one of these dir names (comma-separated); default val eval",
    )
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        default="train",
        help="Skip if path contains any of these dir names (comma-separated)",
    )
    parser.add_argument(
        "--no-dedupe-episode-id",
        action="store_true",
        help="Do not dedupe episode_id (use with --full-vln-ce or when files are known unique)",
    )
    parser.add_argument("--kb-root", type=Path, default=repo_root / "rag4vln" / "data" / "kb" / "memory")
    parser.add_argument("--output-csv", type=Path, default=repo_root / "data" / "vln_ce" / "dataset_gt.csv")
    parser.add_argument("--start-view-root", type=Path, default=repo_root / "data" / "vln_ce" / "start_view")
    parser.add_argument(
        "--end-view-root",
        type=Path,
        default=repo_root / "data" / "vln_ce" / "end_view",
        help="Root for end-view PNGs (written with start render; disable via --no-render-end-view)",
    )
    parser.add_argument(
        "--start-view-subdir-style",
        choices=("mirror_vln_ce", "from_r2r", "mirror_json"),
        default="from_r2r",
        help="Start/end image subdirs: from_r2r=r2r/<split> shared (default, matches eval); see doc for mirror_*",
    )
    parser.add_argument(
        "--no-render-end-view",
        action="store_true",
        help="When start rendering is on, skip end-view PNGs (default renders both)",
    )
    parser.add_argument("--scene-root", type=Path, default=repo_root / "data" / "scene_data" / "mp3d_ce" / "mp3d")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-hfov", type=float, default=90.0)
    parser.add_argument(
        "--camera-height",
        type=float,
        default=KB_AVG_CAMERA_HEIGHT,
        help="Fixed camera height in meters when rendering (written to position[1])",
    )
    parser.add_argument("--no-render-start-view", action="store_true")
    args = parser.parse_args()

    _setup_sys_path(repo_root)
    from src.kb import KnowledgeBase  # noqa: E402

    vln_root = args.vln_root.expanduser().resolve()
    kb_root = args.kb_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    start_view_root = args.start_view_root.expanduser().resolve()
    end_view_root = args.end_view_root.expanduser().resolve()
    scene_root = args.scene_root.expanduser().resolve()
    enable_render = not bool(args.no_render_start_view)
    render_end = enable_render and not bool(args.no_render_end_view)
    subdir_style = str(args.start_view_subdir_style)

    if not vln_root.is_dir():
        raise SystemExit(f"VLN data directory not found: {vln_root}")
    if not kb_root.is_dir():
        raise SystemExit(f"KB directory not found: {kb_root}")

    kb = KnowledgeBase(kb_root)
    if enable_render:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
        from src.utils.habitat_render import make_mp3d_sim, render_rgb_at_pose  # noqa: E402

    kb_scene_set = set(kb.list_scene_ids())
    rows: List[Dict[str, Any]] = []
    seen_episode_keys: Set[Tuple[str, Any]] = set()
    rendered_start = 0
    rendered_end = 0
    render_skipped = 0

    if args.full_vln_ce:
        scan_root = vln_root
        split_names: List[str] = []
        exclude_names: List[str] = []
        dedupe_episode = False
        split_dir_list: List[str] = []
    else:
        scan_root = (vln_root / args.vln_subdir).resolve()
        if not scan_root.is_dir():
            raise SystemExit(f"Scan subdirectory not found: {scan_root}")
        split_names = _split_csv(args.split_dirs)
        exclude_names = _split_csv(args.exclude_dirs)
        dedupe_episode = not bool(args.no_dedupe_episode_id)
        split_dir_list = split_names

    current_scene_for_sim = ""
    current_sim = None

    def _close_sim() -> None:
        nonlocal current_sim, current_scene_for_sim
        if current_sim is not None:
            current_sim.close()
        current_sim = None
        current_scene_for_sim = ""

    def _render_pose_png(
        *,
        view_root: Path,
        rel_file: str,
        ep: Dict[str, Any],
        position: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
    ) -> str:
        nonlocal current_sim, current_scene_for_sim, render_skipped
        raw_scene_id = ep.get("scene_id")
        scene_id = _parse_scene_file_scene_id(raw_scene_id)
        if not scene_id:
            render_skipped += 1
            return ""
        glb = scene_root / scene_id / f"{scene_id}.glb"
        if not glb.is_file():
            render_skipped += 1
            return ""

        rel_sub = _image_subdir_under_vln_view_root(rel_file, subdir_style, repo_root)
        out_dir = view_root / rel_sub
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_id = str(ep.get("episode_id", ""))
        out_path = out_dir / f"ep_{ep_id}.png"

        try:
            if current_sim is None or current_scene_for_sim != scene_id:
                _close_sim()
                current_sim = make_mp3d_sim(
                    glb,
                    int(args.render_width),
                    int(args.render_height),
                    float(args.render_hfov),
                )
                current_scene_for_sim = scene_id
            pos = np.asarray(position, dtype=np.float32)
            pos[1] = float(args.camera_height)
            quat = np.asarray(quat_xyzw, dtype=np.float32)
            rgb = render_rgb_at_pose(current_sim, pos, quat)
            Image.fromarray(rgb).save(out_path, format="PNG")
            return out_path.relative_to(repo_root).as_posix()
        except Exception:
            render_skipped += 1
            return ""

    def _render_start_view(*, rel_file: str, ep: Dict[str, Any]) -> str:
        nonlocal rendered_start
        if not enable_render:
            return ""
        start_pos = _to_xyz(ep.get("start_position"))
        start_rot = _to_quat_xyzw(ep.get("start_rotation"))
        if start_pos is None or start_rot is None:
            render_skipped += 1
            return ""
        rel = _render_pose_png(
            view_root=start_view_root,
            rel_file=rel_file,
            ep=ep,
            position=start_pos,
            quat_xyzw=start_rot,
        )
        if rel:
            rendered_start += 1
        return rel

    def _render_end_view(
        *, rel_file: str, ep: Dict[str, Any], end_pos: Tuple[float, float, float]
    ) -> str:
        nonlocal rendered_end
        if not render_end:
            return ""
        end_rot = _end_rotation_xyzw_from_episode(ep)
        if end_rot is None:
            render_skipped += 1
            return ""
        rel = _render_pose_png(
            view_root=end_view_root,
            rel_file=rel_file,
            ep=ep,
            position=end_pos,
            quat_xyzw=end_rot,
        )
        if rel:
            rendered_end += 1
        return rel

    for file_path in _iter_vln_files(scan_root):
        if split_names and not _file_matches_split_filter(file_path, scan_root, split_names, exclude_names):
            continue
        try:
            data = _load_json(file_path)
        except Exception as e:
            print(f"[warn] read failed, skipping: {file_path} ({e})")
            continue
        episodes = data.get("episodes")
        if not isinstance(episodes, list):
            continue

        try:
            rel_file = file_path.relative_to(repo_root).as_posix()
        except ValueError:
            rel_file = str(file_path)
        try:
            rel_under_scan = file_path.relative_to(scan_root)
        except ValueError:
            rel_under_scan = Path(file_path.name)
        split_tag = _split_tag_from_rel(rel_under_scan, split_dir_list) if split_dir_list else ""

        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            eid = ep.get("episode_id")
            if dedupe_episode and split_tag and eid is not None:
                dedupe_key = (split_tag, eid)
                if dedupe_key in seen_episode_keys:
                    continue
                seen_episode_keys.add(dedupe_key)
            raw_scene_id = ep.get("scene_id")
            gt_scene_id = _parse_kb_scene_id(raw_scene_id)

            start_pos = _to_xyz(ep.get("start_position"))
            end_ref = _pick_end_ref(ep)
            if end_ref is None:
                end_ref = start_pos

            scene_exists = gt_scene_id in kb_scene_set
            gt_start_view_id = ""
            gt_start_zone_id = ""
            gt_start_dist = None
            gt_end_view_id = ""
            gt_end_zone_id = ""
            gt_end_dist = None
            gt_start_view_pos: Optional[Tuple[float, float, float]] = None
            gt_end_view_pos: Optional[Tuple[float, float, float]] = None

            if scene_exists:
                tree = kb.scene(gt_scene_id)
                views = _extract_views(tree)
                gt_start_view_id, gt_start_zone_id, gt_start_view_pos, gt_start_dist = _nearest_view(views, start_pos)
                gt_end_view_id, gt_end_zone_id, gt_end_view_pos, gt_end_dist = _nearest_view(views, end_ref)

            end_img = ""
            if render_end and end_ref is not None:
                end_img = _render_end_view(rel_file=rel_file, ep=ep, end_pos=end_ref)

            row = {
                "episode_id": ep.get("episode_id"),
                "gt_scene_id": gt_scene_id,
                "instruction_text": ((ep.get("instruction") or {}).get("instruction_text") or ""),
                "gt_start_view_id": gt_start_view_id,
                "gt_end_view_id": gt_end_view_id,
                "start_view_image_path": _render_start_view(rel_file=rel_file, ep=ep),
                "end_view_image_path": end_img,
            }
            rows.append(row)

    _close_sim()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id",
        "gt_scene_id",
        "instruction_text",
        "gt_start_view_id",
        "gt_end_view_id",
        "start_view_image_path",
        "end_view_image_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] rows={len(rows)} output={output_csv}")
    if enable_render:
        msg = (
            f"[done] start_view: rendered={rendered_start} root={start_view_root} "
            f"(subdir_style={subdir_style})"
        )
        if render_end:
            msg += f" | end_view: rendered={rendered_end} root={end_view_root}"
        msg += f" | render_skipped={render_skipped}"
        print(msg)


if __name__ == "__main__":
    main()

