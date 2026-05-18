#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 VLN-CE episode 生成 GT 对齐信息（gt_scene / gt_start_view / gt_end_view）。

**终点视角（gt_end_view_id）如何定义**
  1. 在 KB 中取出该场景全部可用 view（含 position、included=True）。
  2. 用 ``_pick_end_ref(episode)`` 得到 3D 参考点：优先 ``reference_path`` 的**最后一个点**，
     否则用 ``goals[0].position``；若皆无则退回 ``start_position``。
  3. ``_nearest_view(views, end_ref)``：在欧氏距离下取 KB 中与该参考点**最近**的 view，作为 GT 终点视角。

默认（评测口径）：只扫描 vln-root 下的 raw_data，且仅 val_seen / val_unseen（排除 train）；
raw_data 内同一 episode 若出现在多个重复 JSON 中，按 episode_id 只保留一行。
mask 子目录（raw_data_mask_*）与 raw_data 轨迹编号一致，默认不扫，避免重复。

默认输出：
  data/vln_ce/dataset_gt.csv
在开启渲染时（默认），会同时生成起点与终点 PNG。默认 ``--start-view-subdir-style from_r2r``：
三套数据（``raw_data`` / ``raw_data_mask_*`` / ``raw_data_implicit``）**共用**同一批图，落在
``data/vln_ce/start_view/r2r/<split>/ep_<id>.png``，与评测脚本读取规则一致。
隐式数据需指定扫描目录，例如 ``--vln-subdir raw_data_implicit``（默认只扫 ``raw_data``）。
仅起点、不渲终点时用 ``--no-render-end-view``；完全不渲图用 ``--no-render-start-view``。

在 InternNav 仓库根目录执行示例：
  python rag4vln/scripts/build_dataset_gt.py
  python rag4vln/scripts/build_dataset_gt.py --vln-subdir raw_data_implicit   # 隐式指令集
  python rag4vln/scripts/build_dataset_gt.py --no-render-end-view   # 只渲起点
  python rag4vln/scripts/build_dataset_gt.py --start-view-subdir-style mirror_vln_ce   # 按 vln_ce 全路径分子目录
  python rag4vln/scripts/build_dataset_gt.py --start-view-subdir-style mirror_json   # 整段相对仓库
  python rag4vln/scripts/build_dataset_gt.py --full-vln-ce   # 整棵 vln_ce
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
    """路径相对 scan_root 的目录名中需命中任一 split，且不得命中任一 exclude。"""
    try:
        rel = file_path.relative_to(scan_root)
    except ValueError:
        return False
    parts = set(rel.parts[:-1])
    if any(x in parts for x in exclude_dir_names):
        return False
    return any(x in parts for x in split_dir_names)


def _split_tag_from_rel(rel: Path, split_dir_names: List[str]) -> str:
    """用于去重：同一 episode_id 在 val_seen 与 val_unseen 可能各有一条（指令/起点不同），须分 split 去重。"""
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
    在 start_view / end_view 根目录下的相对子目录。
    - from_r2r（默认）：自 ``r2r`` 起至 JSON 父目录；三套数据集共用 ``start_view/r2r/<split>/``。
    - mirror_vln_ce：JSON 父目录相对 ``data/vln_ce`` 全路径。
    - mirror_json：``Path(rel_file).parent`` 整段相对仓库（历史嵌套 data/vln_ce/...）。
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
    """绕 Y 轴旋转（弧度），Habitat 常用 x,y,z,w。"""
    half = 0.5 * yaw_rad
    return (0.0, math.sin(half), 0.0, math.cos(half))


def _end_rotation_xyzw_from_episode(ep: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """终点相机朝向：reference_path 末段在 xz 上前进方向；否则用 start_rotation。"""
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
        help="扫描整个 vln-root（含 raw_data_mask_* 等），不做 split 过滤与 episode 去重",
    )
    parser.add_argument(
        "--vln-subdir",
        type=str,
        default="raw_data",
        help="默认仅在此子目录下扫描（相对 vln-root）；与 --full-vln-ce 互斥时以 --full-vln-ce 为准",
    )
    parser.add_argument(
        "--split-dirs",
        type=str,
        default="val_seen,val_unseen",
        help="仅处理路径目录名命中其中任一项的 JSON（逗号分隔）；默认 val 评测",
    )
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        default="train",
        help="路径目录名命中其中任一项则跳过（逗号分隔）",
    )
    parser.add_argument(
        "--no-dedupe-episode-id",
        action="store_true",
        help="不去重 episode_id（仅建议在 --full-vln-ce 或确认无重复文件时使用）",
    )
    parser.add_argument("--kb-root", type=Path, default=repo_root / "rag4vln" / "data" / "kb" / "memory")
    parser.add_argument("--output-csv", type=Path, default=repo_root / "data" / "vln_ce" / "dataset_gt.csv")
    parser.add_argument("--start-view-root", type=Path, default=repo_root / "data" / "vln_ce" / "start_view")
    parser.add_argument(
        "--end-view-root",
        type=Path,
        default=repo_root / "data" / "vln_ce" / "end_view",
        help="终点视角 PNG 根目录（与起点同时渲染时写入；可用 --no-render-end-view 关闭终点）",
    )
    parser.add_argument(
        "--start-view-subdir-style",
        choices=("mirror_vln_ce", "from_r2r", "mirror_json"),
        default="from_r2r",
        help="起点/终点图子目录：from_r2r=r2r/<split> 共用图（默认，与 eval 一致）；mirror_vln_ce / mirror_json 见脚本说明",
    )
    parser.add_argument(
        "--no-render-end-view",
        action="store_true",
        help="在已开启起点渲染时，跳过终点视角 PNG（默认会渲终点）",
    )
    parser.add_argument("--scene-root", type=Path, default=repo_root / "data" / "scene_data" / "mp3d_ce" / "mp3d")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-hfov", type=float, default=90.0)
    parser.add_argument(
        "--camera-height",
        type=float,
        default=KB_AVG_CAMERA_HEIGHT,
        help="渲染时使用的固定相机高度（米，写入 position[1]）",
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
        raise SystemExit(f"VLN 数据目录不存在: {vln_root}")
    if not kb_root.is_dir():
        raise SystemExit(f"KB 目录不存在: {kb_root}")

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
            raise SystemExit(f"扫描子目录不存在: {scan_root}")
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
            print(f"[warn] 读取失败，跳过: {file_path} ({e})")
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

