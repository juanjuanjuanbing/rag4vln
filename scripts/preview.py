#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出所有场景的 BEV 预览图（真实加载 3D 场景，不使用 KB 视角图）。

功能：
1) 从 KB 读取 scene_id 列表；
2) 在 ``scenes_dir`` 中定位对应 glb；
3) 使用 Habitat-Sim 渲染俯视图，保存到 ``<out_dir>/scenes/``；
   - ``pinhole``：透视投影，构图按 navmesh；
   - ``orthographic``：正射全景：navmesh 与场景网格根 ``cumulative_bb`` 取并集后构图，尽量包住整栋几何；
4) 生成总览拼图 ``<out_dir>/overview.png``（透明背景）；
5) 写失败清单 ``<out_dir>/missing_scenes.txt``（找不到 glb 或渲染失败）。

默认输出目录：``rag4vln/result/preview``（按用户要求）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


def _setup_sys_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rag4vln_root = repo_root / "rag4vln"
    if str(rag4vln_root) not in sys.path:
        sys.path.insert(0, str(rag4vln_root))


def _safe_name(scene_id: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in scene_id)


def _candidate_glb_paths(scenes_dir: Path, scene_id: str) -> List[Path]:
    """兼容几种常见 MP3D 目录布局。"""
    cands = [
        scenes_dir / scene_id / f"{scene_id}.glb",
        scenes_dir / scene_id / f"{scene_id}.basis.glb",
        scenes_dir / "mp3d" / scene_id / f"{scene_id}.glb",
        scenes_dir / "mp3d" / scene_id / f"{scene_id}.basis.glb",
        scenes_dir / f"{scene_id}.glb",
        scenes_dir / f"{scene_id}.basis.glb",
    ]
    return cands


def _resolve_scene_glb(scenes_dir: Path, scene_id: str) -> Optional[Path]:
    for p in _candidate_glb_paths(scenes_dir, scene_id):
        if p.is_file():
            return p
    return None


def _cumulative_bb_min_max(sim: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """场景图根节点累计 AABB（含 glb 内全部 drawable），失败返回 None。"""
    try:
        bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    except Exception:
        return None
    try:
        mn = bb.min
        mx = bb.max
        bmin = np.array([float(mn.x), float(mn.y), float(mn.z)], dtype=np.float32)
        bmax = np.array([float(mx.x), float(mx.y), float(mx.z)], dtype=np.float32)
        return bmin, bmax
    except Exception:
        pass
    try:
        bmin = np.asarray(bb.min, dtype=np.float32).reshape(3)
        bmax = np.asarray(bb.max, dtype=np.float32).reshape(3)
        return bmin, bmax
    except Exception:
        return None


def _union_aabb(
    a: tuple[np.ndarray, np.ndarray],
    b: Optional[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if b is None:
        return a
    amin, amax = a
    bmin, bmax = b
    return np.minimum(amin, bmin), np.maximum(amax, bmax)


def _preview_bounds_full_scene(scene_glb: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    navmesh 与场景网格根 AABB 的并集。

    仅供正射「全景」：navmesh 常小于建筑外墙/台阶等几何；仅用 navmesh 会裁掉场景。
    """
    import habitat_sim
    from habitat_sim import SimulatorConfiguration
    from habitat_sim.agent import AgentConfiguration
    from habitat_sim.sensor import CameraSensorSpec, SensorType

    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_glb.resolve())
    spec = CameraSensorSpec()
    spec.uuid = "rgb"
    spec.sensor_type = SensorType.COLOR
    spec.resolution = [32, 32]
    spec.hfov = 90.0
    spec.near = 0.05
    spec.far = 2000.0
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    simulator = habitat_sim.Simulator(cfg)
    try:
        pf = simulator.pathfinder
        if not pf.is_loaded:
            raise RuntimeError("pathfinder is not loaded (navmesh unavailable)")
        nmin, nmax = pf.get_bounds()
        nav = (np.asarray(nmin, dtype=np.float32), np.asarray(nmax, dtype=np.float32))
        mesh = _cumulative_bb_min_max(simulator)
        return _union_aabb(nav, mesh)
    finally:
        simulator.close()


def _ortho_scale_fit_horizontal_scene(
    *,
    dx: float,
    dz: float,
    width: int,
    height: int,
    margin: float,
    slack: float,
    expand: float,
) -> float:
    """
    正射俯视：使轴对齐水平矩形 ``dx × dz``（加 margin、slack）落在画面内。

    Habitat C++：``orthographicProjection(nearPlaneSize / orthoScale, ...)``，
    ``nearPlaneSize=(1, aspect)``（aspect 来自 resolution，实现上可能与 W/H 或 H/W 一致）。
    orthoScale **越小** → 画面里看到的水平范围 **越大**。

    同时对 W/H 与 H/W 两种 aspect 约定取约束的并集（偏保守，多留出视野）。
    """
    pad = (1.0 + max(0.0, float(margin))) * max(1.0, float(slack)) * max(1.0, float(expand))
    dx_m = max(float(dx), 1e-3) * pad
    dz_m = max(float(dz), 1e-3) * pad
    wh = float(width) / max(float(height), 1.0)
    hw = float(height) / max(float(width), 1.0)
    ortho = min(
        2.0 / dx_m,
        2.0 / dz_m,
        2.0 * wh / dx_m,
        2.0 * wh / dz_m,
        2.0 * hw / dx_m,
        2.0 * hw / dz_m,
    )
    return float(max(1e-4, min(500.0, ortho)))


def _ortho_camera_height_y(bmin: np.ndarray, bmax: np.ndarray) -> float:
    """
    正射俯视相机世界坐标 Y（米）。

    说明：**平行投影下，相机抬高不会改变画面里水平方向「装进多少户型」**，
    那是 ``ortho_scale`` + 包围盒 dx/dz 决定的。高度只用于：
    - 落在整栋 ``bmax.y`` 之上，避免穿模；
    - 离房顶足够远，减轻 near 裁剪与 Z-fighting；
    - 与 ``far`` 一起盖住从房顶到地面的深度范围。
    """
    dx = float(bmax[0] - bmin[0])
    dy = float(bmax[1] - bmin[1])
    dz = float(bmax[2] - bmin[2])
    ext_xz = max(dx, dz, 1e-3)
    # 房顶之上的净空：与楼高、占地挂钩，并设下限（米）
    clear_above_roof = max(dy * 0.55, ext_xz * 0.22, 22.0)
    return float(bmax[1] + clear_above_roof)


def _render_bev(
    scene_glb: Path,
    width: int,
    height: int,
    hfov: float,
    *,
    projection: str = "pinhole",
    ortho_scale: Optional[float] = None,
    ortho_margin: float = 0.04,
    ortho_slack: float = 1.03,
    ortho_expand: float = 1.03,
    ortho_auto_scale: float = 1.03,
    ortho_debug: bool = False,
) -> np.ndarray:
    """
    真实加载场景并渲染俯视 RGB（不含 alpha；透明在保存前单独处理）。
    透视：navmesh bounds；正射全景：navmesh 与场景根 ``cumulative_bb`` 并集 + 正射比例适配。

    projection:
      - ``pinhole``：透视（原行为）；
      - ``orthographic``：正射，无透视变形（Habitat ``SensorSubType.ORTHOGRAPHIC``）。
    """
    import habitat_sim
    from habitat_sim import SimulatorConfiguration
    from habitat_sim.agent import AgentConfiguration
    from habitat_sim.sensor import CameraSensorSpec, SensorType

    proj = str(projection or "pinhole").strip().lower()
    if proj not in ("pinhole", "orthographic"):
        raise ValueError(f"projection must be pinhole|orthographic, got {projection!r}")

    # 正射：需先读 bounds 才能在创建 Simulator 前设置 ortho_scale（多打开一次小分辨率 sim）。
    # 透视：保持单开 sim，与旧版一致。
    ortho_scale_v: Optional[float] = None
    top_y_pre: Optional[float] = None
    far_plane = 2000.0
    if proj == "orthographic":
        bpre_min, bpre_max = _preview_bounds_full_scene(scene_glb)
        dx_pre = float(bpre_max[0] - bpre_min[0])
        dy_pre = float(bpre_max[1] - bpre_min[1])
        dz_pre = float(bpre_max[2] - bpre_min[2])
        extent_pre = float(max(dx_pre, dz_pre))
        if extent_pre <= 1e-3:
            extent_pre = 5.0
            dx_pre = dz_pre = extent_pre
        top_y_pre = _ortho_camera_height_y(bpre_min, bpre_max)
        if ortho_scale is not None:
            ortho_scale_v = float(ortho_scale)
        else:
            ortho_scale_v = _ortho_scale_fit_horizontal_scene(
                dx=dx_pre,
                dz=dz_pre,
                width=int(width),
                height=int(height),
                margin=float(ortho_margin),
                slack=float(ortho_slack),
                expand=float(ortho_expand),
            )
            # 自动 ortho 之后再乘系数：<1 视野更大（易裁边），>1 略收紧（场景更满，贴边只留一点）
            ortho_scale_v *= float(max(0.45, min(1.18, ortho_auto_scale)))
        # 向下看到地面以下也要在 far 内：相机到最低点 + 余量
        # 略夸大 far：第二次打开 sim 时并集包围盒可能略胀，避免远截面切掉地面
        far_plane = float(
            max(12000.0, float(top_y_pre) - float(bpre_min[1]) + max(dy_pre, 150.0) + 900.0)
        )
        if ortho_debug:
            print(
                f"[preview ortho] scene_glb={scene_glb.name} "
                f"aabb dx={dx_pre:.2f} dy={dy_pre:.2f} dz={dz_pre:.2f} "
                f"ortho_scale={float(ortho_scale_v):.6f} top_y={float(top_y_pre):.2f} "
                f"far={far_plane:.1f} (clear_above_roof={float(top_y_pre - bpre_max[1]):.2f})",
                flush=True,
            )

    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_glb.resolve())

    spec = CameraSensorSpec()
    spec.uuid = "rgb"
    spec.sensor_type = SensorType.COLOR
    spec.resolution = [height, width]
    spec.hfov = float(hfov)
    spec.near = 0.01 if proj == "orthographic" else 0.05
    spec.far = far_plane
    # 俯视：绕 X 轴 -90°（弧度）
    spec.orientation = [-math.pi / 2.0, 0.0, 0.0]
    spec.position = [0.0, 0.0, 0.0]
    if proj == "orthographic":
        spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
        spec.ortho_scale = float(ortho_scale_v)  # type: ignore[assignment]
    else:
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    try:
        pf = sim.pathfinder
        if not pf.is_loaded:
            raise RuntimeError("pathfinder is not loaded (navmesh unavailable)")
        if proj == "pinhole":
            bmin, bmax = pf.get_bounds()
            bmin = np.asarray(bmin, dtype=np.float32)
            bmax = np.asarray(bmax, dtype=np.float32)
            center = (bmin + bmax) / 2.0
            extent_xz = float(max(bmax[0] - bmin[0], bmax[2] - bmin[2]))
            if extent_xz <= 1e-3:
                extent_xz = 5.0
            half_fov_rad = math.radians(float(hfov) / 2.0)
            if half_fov_rad <= 1e-6:
                half_fov_rad = math.radians(45.0)
            fit_height = (extent_xz * 0.5) / max(math.tan(half_fov_rad), 1e-6)
            top_y = float(bmax[1] + fit_height * 0.72 + 0.8)
        else:
            nmin, nmax = pf.get_bounds()
            nav = (np.asarray(nmin, dtype=np.float32), np.asarray(nmax, dtype=np.float32))
            bmin, bmax = _union_aabb(nav, _cumulative_bb_min_max(sim))
            center = (bmin + bmax) / 2.0
            top_y = _ortho_camera_height_y(bmin, bmax)
            if ortho_debug:
                dx_ = float(bmax[0] - bmin[0])
                dz_ = float(bmax[2] - bmin[2])
                print(
                    f"[preview ortho] render-pass aabb dx={dx_:.2f} dz={dz_:.2f} top_y={top_y:.2f}",
                    flush=True,
                )

        state = habitat_sim.AgentState()
        state.position = np.array([center[0], top_y, center[2]], dtype=np.float32)
        sim.get_agent(0).set_state(state)
        obs = sim.get_sensor_observations()["rgb"]
        rgb = obs[..., :3] if obs.shape[-1] == 4 else obs
        rgb = np.asarray(rgb, dtype=np.uint8)

        # 透视：自动去黑边放大主体。正射全图时不要裁切，否则会丢掉外墙/留白。
        if proj == "pinhole":
            gray = rgb.mean(axis=2)
            mask = gray > 4.0
            ys, xs = np.where(mask)
            if ys.size > 0 and xs.size > 0:
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                h = max(1, y1 - y0 + 1)
                w = max(1, x1 - x0 + 1)
                pad_y = max(4, int(h * 0.03))
                pad_x = max(4, int(w * 0.03))
                y0 = max(0, y0 - pad_y)
                y1 = min(rgb.shape[0] - 1, y1 + pad_y)
                x0 = max(0, x0 - pad_x)
                x1 = min(rgb.shape[1] - 1, x1 + pad_x)
                crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
                if crop.size > 0 and (crop.shape[0] < rgb.shape[0] or crop.shape[1] < rgb.shape[1]):
                    from PIL import Image

                    rgb = np.asarray(
                        Image.fromarray(crop).resize((width, height), resample=Image.Resampling.LANCZOS),
                        dtype=np.uint8,
                    )
        return rgb
    finally:
        sim.close()


def _filter_scene_ids(scene_ids: List[str], only_patterns: Optional[List[str]]) -> List[str]:
    """按 ``--only-scenes`` 筛选：先精确匹配，再唯一前缀，再唯一子串。"""
    if not only_patterns:
        return list(scene_ids)
    picked: List[str] = []
    for raw in only_patterns:
        pat = str(raw).strip()
        if not pat:
            continue
        exact = [s for s in scene_ids if s == pat]
        if exact:
            picked.extend(exact)
            continue
        pref = [s for s in scene_ids if s.startswith(pat)]
        if len(pref) == 1:
            picked.append(pref[0])
            continue
        if len(pref) > 1:
            raise SystemExit(f"[preview] 场景前缀 {pat!r} 对应多项（请写全 id）：{pref}")
        sub = [s for s in scene_ids if pat in s]
        if len(sub) == 1:
            picked.append(sub[0])
            continue
        if len(sub) > 1:
            raise SystemExit(f"[preview] 子串 {pat!r} 对应多项（请写更长的前缀）：{sub}")
        raise SystemExit(f"[preview] 未找到匹配场景 {pat!r}")
    # 去重保序
    seen: set[str] = set()
    out: List[str] = []
    for s in picked:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# 常见抠像绿（偏广电绿，避免与植被纯 #00FF00 完全重叠时可再调）
_GREEN_SCREEN_RGB = np.array([0, 177, 64], dtype=np.uint8)


def _rgb_to_rgba(rgb: np.ndarray, bg_threshold: int, *, background: str = "transparent") -> np.ndarray:
    """
    按 ``bg_threshold`` 区分前景与背景。

    - ``transparent``：背景 alpha=0；
    - ``greenscreen``：背景铺绿幕 RGB，alpha=255（便于后期抠图）。
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    gray = rgb.astype(np.float32).mean(axis=2)
    mx = rgb.max(axis=2).astype(np.float32)
    thr = float(max(0, int(bg_threshold)))
    fg = (gray > thr) | (mx > thr + 2.0)
    h, w = rgb.shape[0], rgb.shape[1]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    mode = str(background or "transparent").strip().lower()
    if mode == "greenscreen":
        bg_col = _GREEN_SCREEN_RGB.reshape(1, 1, 3)
        out[..., :3] = np.where(fg[..., np.newaxis], rgb, bg_col)
        out[..., 3] = 255
    else:
        out[..., :3] = rgb
        out[..., 3] = (fg.astype(np.uint8) * 255)
    return out


def _build_overview(
    image_files: List[Path],
    out_path: Path,
    *,
    tile_w: int = 320,
    tile_h: int = 240,
    pad: int = 12,
    canvas_rgba: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> None:
    from PIL import Image, ImageOps, ImageDraw

    if not image_files:
        return
    n = len(image_files)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad
    canvas = Image.new("RGBA", (canvas_w, canvas_h), canvas_rgba)
    draw = ImageDraw.Draw(canvas)

    for idx, p in enumerate(image_files):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)
        im = Image.open(p).convert("RGBA")
        tile = ImageOps.fit(im, (tile_w, tile_h), method=Image.Resampling.LANCZOS)
        canvas.paste(tile, (x0, y0), tile)
        # 简短标注（深色字 + 白描边，在透明底上可读）
        label = p.stem[:28]
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            draw.text((x0 + 6 + dx, y0 + 6 + dy), label, fill=(255, 255, 255, 255))
        draw.text((x0 + 6, y0 + 6), label, fill=(30, 30, 30, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scene preview images from rag4vln KB")
    parser.add_argument(
        "--kb-root",
        type=Path,
        default=Path("rag4vln/data/kb/memory"),
        help="KB 根目录（包含 manifest.json / scenes / imgs）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("rag4vln/result/preview"),
        help="输出目录（默认 rag4vln/result/preview）",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path("data/scene_data/mp3d_ce"),
        help="Habitat 场景根目录（包含各 scene 的 glb）",
    )
    parser.add_argument("--width", type=int, default=1024, help="单张 BEV 宽度")
    parser.add_argument("--height", type=int, default=1024, help="单张 BEV 高度")
    parser.add_argument("--hfov", type=float, default=120.0, help="俯视相机水平视场角（透视时参与构图；正射时用于自动估计 ortho_scale）")
    parser.add_argument(
        "--projection",
        choices=("pinhole", "orthographic"),
        default="pinhole",
        help="pinhole=透视俯视（默认）；orthographic=正射俯视，无透视",
    )
    parser.add_argument(
        "--ortho-scale",
        type=float,
        default=None,
        help="仅 orthographic：Habitat ortho_scale，越大画面越“放大”（视野越小）。不设则按 navmesh 全包围盒自动算",
    )
    parser.add_argument(
        "--ortho-margin",
        type=float,
        default=0.04,
        help="仅 orthographic 且未指定 --ortho-scale：相对水平包围盒的放大比例，略大于 0 即「比场景大一点点」防裁边",
    )
    parser.add_argument(
        "--ortho-slack",
        type=float,
        default=1.03,
        help="仅 orthographic 且未指定 --ortho-scale：在 margin 上再乘的细调（略大于 1 即可，过大留白多）",
    )
    parser.add_argument(
        "--ortho-expand",
        type=float,
        default=1.03,
        help="仅 orthographic 且未指定 --ortho-scale：第三层乘子，默认贴近「刚好包住」",
    )
    parser.add_argument(
        "--ortho-auto-scale",
        type=float,
        default=1.03,
        help="仅 orthographic 且未指定 --ortho-scale：自动 ortho 之后再乘；默认略 >1 让场景只占画面绝大部分、边距很小。<1 拉远，>1 拉近",
    )
    parser.add_argument(
        "--ortho-debug",
        action="store_true",
        help="打印正射用的包围盒尺寸、ortho_scale、相机高度 top_y、far 等（核对「全景」构图）",
    )
    parser.add_argument(
        "--only-scenes",
        nargs="+",
        default=None,
        metavar="ID",
        help="只渲染列出的 scene_id；可与 KB 中 id 精确匹配，或为唯一前缀/子串（如 2t7W → 2t7WUuJeko7）",
    )
    parser.add_argument(
        "--bg-threshold",
        type=int,
        default=18,
        help="区分前景/背景的亮度阈值（0-255）；透明底或绿幕均使用该阈值",
    )
    parser.add_argument(
        "--background",
        choices=("transparent", "greenscreen"),
        default="transparent",
        help="透明 PNG（默认）或绿幕 RGB 背景（抠像用，绿色约为 0,177,64）",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    _setup_sys_path(repo_root)
    kb_root = args.kb_root if args.kb_root.is_absolute() else (repo_root / args.kb_root).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (repo_root / args.out_dir).resolve()
    scenes_dir = args.scenes_dir if args.scenes_dir.is_absolute() else (repo_root / args.scenes_dir).resolve()
    scenes_out = out_dir / "scenes"

    from src.kb import KnowledgeBase  # noqa: E402

    if not kb_root.is_dir():
        raise SystemExit(f"KB root not found: {kb_root}")
    if not scenes_dir.is_dir():
        raise SystemExit(f"Scenes dir not found: {scenes_dir}")

    kb = KnowledgeBase(kb_root)
    scene_ids = kb.list_scene_ids()
    if not scene_ids:
        raise SystemExit(f"No scenes in KB: {kb_root}")
    try:
        scene_ids = _filter_scene_ids(scene_ids, args.only_scenes)
    except SystemExit:
        raise
    if args.only_scenes:
        print(f"[preview] only-scenes → {len(scene_ids)} scene(s): {scene_ids}", flush=True)

    from PIL import Image  # noqa: E402

    exported: List[Path] = []
    missed: List[str] = []
    bg_thr = max(0, min(255, int(args.bg_threshold)))
    stem_suffix = "" if args.projection == "pinhole" else "_ortho"
    for sid in scene_ids:
        out_file = scenes_out / f"{_safe_name(sid)}{stem_suffix}.png"
        glb = _resolve_scene_glb(scenes_dir, sid)
        if glb is None:
            missed.append(f"{sid}\tmissing_glb")
            continue
        try:
            rgb = _render_bev(
                glb,
                width=int(args.width),
                height=int(args.height),
                hfov=float(args.hfov),
                projection=str(args.projection),
                ortho_scale=args.ortho_scale,
                ortho_margin=float(args.ortho_margin),
                ortho_slack=float(args.ortho_slack),
                ortho_expand=float(args.ortho_expand),
                ortho_auto_scale=float(args.ortho_auto_scale),
                ortho_debug=bool(args.ortho_debug),
            )
            rgba = _rgb_to_rgba(rgb, bg_thr, background=str(args.background))
            out_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, mode="RGBA").save(out_file)
            exported.append(out_file)
        except Exception as e:
            missed.append(f"{sid}\trender_error\t{type(e).__name__}: {e}")

    overview_name = "overview.png" if args.projection == "pinhole" else "overview_ortho.png"
    ov_bg: tuple[int, int, int, int] = (
        (0, 177, 64, 255) if args.background == "greenscreen" else (0, 0, 0, 0)
    )
    _build_overview(exported, out_dir / overview_name, canvas_rgba=ov_bg)
    if missed:
        (out_dir / "missing_scenes.txt").write_text("\n".join(missed) + "\n", encoding="utf-8")
    print(
        f"[preview] scenes_total={len(scene_ids)} exported={len(exported)} missed={len(missed)} "
        f"projection={args.projection}"
    )
    print(f"[preview] scenes_dir={scenes_dir}")
    print(f"[preview] output_dir={out_dir}")


if __name__ == "__main__":
    main()

