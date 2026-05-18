# -*- coding: utf-8 -*-
"""
rag4vln 插入 InternVLA（HabitatVLN）的评测入口。

设计目标：
- 模仿 `internnav/scripts/eval/eval.py` 的最小接口：`--config <eval_cfg.py>`
- 但在评测前：对 dataset 里的 `instruction.instruction_text` 做一次
  “rag4vln 检索 + 指令增强”，把增强后的指令替换进去，再跑评测。

默认策略（为了可重复 + 避免 progress.json 跳过）：
- 生成临时 eval cfg，强制把 `eval_settings.output_path` 指向一个新目录

视角图导出与 ``eval_retriever.py`` 一致：默认写入本次 run 目录；加 ``--no-export-images`` 则关闭。
写入 ``ins_start_view/``、``retriever_start_view/``、``retriever_end_view/``；若存在 ``--gt-csv`` 则额外写入 ``gt_start_view/``、``gt_end_view/``。

运行提示：若用 ``conda run`` 且长时间看不到终端输出，请加 ``--no-capture-output``
（否则会缓冲到进程结束才打印）。见 ``rag4vln/scripts/eval/README.md``。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import importlib.util
import json
import os
import shutil
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml  # type: ignore
from tqdm import tqdm  # type: ignore


def _setup_sys_path(repo_root: Path) -> None:
    # 让 `import internnav.*` 生效
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rag4vln_root = repo_root / "rag4vln"
    if str(rag4vln_root) not in sys.path:
        sys.path.insert(0, str(rag4vln_root))


def _load_eval_cfg_py(eval_cfg_py: Path) -> Any:
    spec = importlib.util.spec_from_file_location("eval_cfg_module", str(eval_cfg_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load eval cfg: {eval_cfg_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_cfg_module"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "eval_cfg"):
        raise RuntimeError(f"No `eval_cfg` found in {eval_cfg_py}")
    return getattr(mod, "eval_cfg")


def _find_data_path_from_habitat_yaml(habitat_yaml_path: Path) -> Tuple[str, str, str]:
    cfg = yaml.safe_load(habitat_yaml_path.read_text(encoding="utf-8"))
    habitat = cfg.get("habitat") or {}
    dataset = habitat.get("dataset") or {}
    data_path = dataset.get("data_path")
    split = dataset.get("split")
    scenes_dir = dataset.get("scenes_dir")
    if not isinstance(data_path, str) or not isinstance(split, str) or not isinstance(scenes_dir, str):
        raise RuntimeError(f"Unexpected habitat dataset config in {habitat_yaml_path}")
    return data_path, split, scenes_dir


def _resolve_dataset_gz(repo_root: Path, data_path_pattern: str, split: str) -> Path:
    dataset_filename = data_path_pattern.format(split=split)
    p = Path(dataset_filename)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _patch_eval_cfg_output_path(eval_cfg_py: Path, new_output_path: str) -> Path:
    """
    在原 eval cfg python 上原地替换 eval_settings.output_path（不再另存一份超长文件名副本）。
    """
    text = eval_cfg_py.read_text(encoding="utf-8")
    # 只替换第一个 output_path: "..."
    pattern = r'("output_path"\s*:\s*")([^"]+)(")'

    def repl(m: re.Match) -> str:
        return m.group(1) + new_output_path + m.group(3)

    text2, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n < 1:
        raise RuntimeError(f"Failed to patch output_path in eval cfg: {eval_cfg_py}")

    eval_cfg_py.write_text(text2, encoding="utf-8")
    return eval_cfg_py


def _extract_path_region_descriptions(kb: Any, plan: dict) -> List[str]:
    pairs = plan.get("topk3_pairs") or []
    if not pairs or not isinstance(pairs[0], dict):
        return []
    pair = pairs[0]
    scene_id = pair.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        return []
    tree = kb.scene(scene_id)
    zones = tree.get("zones") or {}
    if not isinstance(zones, dict):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for zid in pair.get("path_zone_ids") or []:
        if not isinstance(zid, str) or not zid or zid in seen:
            continue
        seen.add(zid)
        zdesc = ((zones.get(zid) or {}).get("attributes") or {}).get("description")
        s = str(zdesc).strip() if zdesc is not None else ""
        out.append(s if s else zid)
    return out


def _build_augmenter(
    augmenter_kind: str,
    augment_cfg_path: Optional[Path],
) -> Any:
    from src.augment import (  # noqa: E402
        build_llm_direct_augmenter,
        build_r_only_augmenter,
        build_semantic_pathplanning_augmenter,
        build_template_path_augmenter,
    )

    cfg = augment_cfg_path if augment_cfg_path is not None and augment_cfg_path.is_file() else None
    if augmenter_kind == "r_only":
        return build_r_only_augmenter(config_path=cfg)
    if augmenter_kind == "semantic_pathplanning":
        return build_semantic_pathplanning_augmenter(config_path=cfg)
    if augmenter_kind == "template_path":
        return build_template_path_augmenter(config_path=cfg)
    return build_llm_direct_augmenter(config_path=cfg)


def _cfg_embedding_dim(config_path: Path) -> int:
    from src.config_io import retrieval_embedding_dim  # noqa: E402

    try:
        return retrieval_embedding_dim(config_path)
    except Exception:
        return 768


def _load_robot_image(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    return Image.open(path).convert("RGB")


def _resolve_start_view_png(
    repo_root: Path,
    start_view_root: Path,
    ep_id: str,
    dataset_file: Path,
) -> Path:
    """共用 ``start_view/r2r/<split>/ep_<id>.png``，与 eval_retriever / build_dataset_gt 的 from_r2r 一致。"""
    vln_ce_root = (repo_root / "data" / "vln_ce").resolve()
    ds_parent = dataset_file.expanduser().resolve().parent
    try:
        rel_parts = ds_parent.relative_to(vln_ce_root).parts
    except ValueError as e:
        raise SystemExit(
            f"start_view 路径要求 dataset JSON 位于 data/vln_ce 下: {dataset_file}"
        ) from e
    try:
        i = rel_parts.index("r2r")
    except ValueError as e:
        raise SystemExit(
            f"start_view 共用路径要求在 vln_ce 相对目录中包含 r2r 段: {dataset_file}"
        ) from e
    sub = Path(*rel_parts[i:])
    return (start_view_root / sub / f"ep_{ep_id}.png").resolve()


def _parse_kb_scene_id(vln_scene_id: Any) -> str:
    """与 ``build_dataset_gt._parse_kb_scene_id`` 一致：episode.scene_id → KB 场景名。"""
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


def _load_gt_rows_by_episode(gt_csv_path: Path) -> Dict[str, List[Dict[str, str]]]:
    """``dataset_gt.csv`` 中同一 ``episode_id`` 可能在 val_seen/val_unseen 各一行，须保留列表。"""
    out: Dict[str, List[Dict[str, str]]] = {}
    with gt_csv_path.open(encoding="utf-8", newline="") as gf:
        for row in csv.DictReader(gf):
            eid = str(row.get("episode_id", "")).strip()
            if not eid:
                continue
            out.setdefault(eid, []).append(
                {
                    "gt_scene_id": str(row.get("gt_scene_id", "")).strip(),
                    "gt_start_view_id": str(row.get("gt_start_view_id", "")).strip(),
                    "gt_end_view_id": str(row.get("gt_end_view_id", "")).strip(),
                }
            )
    return out


def _lookup_gt_row(
    gt_rows_by_ep: Dict[str, List[Dict[str, str]]],
    ep_id: str,
    episode_scene_id: Any,
) -> Optional[Dict[str, str]]:
    rows = gt_rows_by_ep.get(ep_id) or []
    if not rows:
        return None
    kb_scene = _parse_kb_scene_id(episode_scene_id)
    if kb_scene:
        for row in rows:
            if row.get("gt_scene_id") == kb_scene:
                return row
    if len(rows) == 1:
        return rows[0]
    return None


def _save_kb_view_image(kb: Any, scene_id: str, view_id: str, dest: Path) -> bool:
    """与 ``eval_retriever.py`` 一致：优先拷贝 KB 磁盘文件，否则 PIL 保存。"""
    p = kb.view_image_path(scene_id, view_id)
    if p is not None and p.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        return True
    img = kb.load_view_image(scene_id, view_id)
    if img is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return True
    return False


def _export_retriever_rank_view_images(
    kb: Any,
    plan: Dict[str, Any],
    export_root: Path,
    ep_id: str,
    k_export: int,
) -> None:
    """导出检索 ``topk3_pairs`` 前 k 名的 KB 起/终点视角图（与检索器评测目录名一致）。"""
    top_pairs = plan.get("topk3_pairs") or []
    for rank, pair in enumerate(top_pairs[: max(1, k_export)], start=1):
        if not isinstance(pair, dict):
            continue
        rsid = str(pair.get("scene_id") or "").strip()
        sv = str(pair.get("start_view_id") or "").strip()
        ev = str(pair.get("end_view_id") or "").strip()
        rel_s = f"retriever_start_view/ep_{ep_id}_{rank:02d}.png"
        rel_e = f"retriever_end_view/ep_{ep_id}_{rank:02d}.png"
        if rsid and sv:
            if not _save_kb_view_image(kb, rsid, sv, export_root / rel_s):
                print(
                    f"[export] kb view missing retriever_start ep={ep_id} rank={rank} scene={rsid} view={sv}",
                    flush=True,
                )
        if rsid and ev:
            if not _save_kb_view_image(kb, rsid, ev, export_root / rel_e):
                print(
                    f"[export] kb view missing retriever_end ep={ep_id} rank={rank} scene={rsid} view={ev}",
                    flush=True,
                )


def _export_ins_start_view(export_root: Path, ep_id: str, ins_src: Optional[Path]) -> None:
    """``ins_start_view``：与检索输入一致的起始观测 PNG（与 eval_retriever 一致）。"""
    if ins_src is not None and ins_src.is_file():
        dest_ins = export_root / "ins_start_view" / f"ep_{ep_id}.png"
        dest_ins.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ins_src, dest_ins)


def _export_gt_kb_endpoint_views(
    kb: Any,
    export_root: Path,
    ep_id: str,
    gt_scene: str,
    gt_start_view: str,
    gt_end_view: str,
) -> None:
    """``gt_start_view`` / ``gt_end_view``：GT CSV 对应 KB 全景（检索器评测目录无此两项，作对照用）。"""
    if gt_scene and gt_start_view:
        p = export_root / "gt_start_view" / f"ep_{ep_id}.png"
        if not _save_kb_view_image(kb, gt_scene, gt_start_view, p):
            print(
                f"[export] kb gt_start missing ep={ep_id} scene={gt_scene} view={gt_start_view}",
                flush=True,
            )
    if gt_scene and gt_end_view:
        p = export_root / "gt_end_view" / f"ep_{ep_id}.png"
        if not _save_kb_view_image(kb, gt_scene, gt_end_view, p):
            print(
                f"[export] kb gt_end missing ep={ep_id} scene={gt_scene} view={gt_end_view}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="rag4vln augmented VLN eval entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "提示：使用 `conda run` 时默认会捕获 stdout/stderr，长时间跑可能像「没有任何输出」。\n"
            "请加上 `--no-capture-output`（或 `--live-stream`）以便实时看到日志，例如：\n"
            "  conda run --no-capture-output -n inter_hab python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py ...\n"
            "多行命令里反斜杠续行必须是行末最后一个字符；`--save-instruction-pairs` 不要拆成两行。\n"
            "离线或 DNS 失败（OpenAI APIConnectionError）：请加 `--no-robot-image`，检索阶段将不调 VLM caption。"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py",
        help="InternNav 的 eval cfg python（相对 repo_root 或绝对路径）",
    )
    parser.add_argument(
        "--augmenter",
        choices=("llm_direct", "template_path", "semantic_pathplanning", "r_only"),
        default="semantic_pathplanning",
        help="含 r_only：仅拼接检索证据与原始指令（无 LLM baseline）",
    )
    parser.add_argument(
        "--rag4vln-config",
        type=Path,
        default=Path("rag4vln/src/config.yaml"),
        help="rag4vln 统一 YAML（basic / retrieval / augment）",
    )
    parser.add_argument("--kb-root", type=Path, default=Path("rag4vln/data/kb/memory"))
    parser.add_argument("--text-embedder", choices=("auto", "bert", "sbert", "bge", "binary"), default="binary")
    parser.add_argument("--vision-embedder", choices=("vit", "binary"), default="binary")
    parser.add_argument("--binary-dim", type=int, default=64)
    parser.add_argument("--topk1", type=int, default=3)
    parser.add_argument("--topk2", type=int, default=3)
    parser.add_argument("--topk3", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=1, help="只增强前 N 个 episode，评测也只跑这 N 个")
    parser.add_argument("--robot-image", type=Optional[Path], default=None, help="可选：固定使用该图作为起始图像")
    parser.add_argument(
        "--no-robot-image",
        action="store_true",
        help="检索不传起始观测：跳过 VLM caption（无 OpenAI 网络调用），机器人图文查询嵌入为零；可与 r_only 等搭配离线跑",
    )
    parser.add_argument(
        "--kb-embed-cache",
        type=Path,
        default=None,
        help="KB embedding 缓存文件（.pt）；首次构建后可复用以显著提速",
    )
    parser.add_argument(
        "--rebuild-kb-embed-cache",
        action="store_true",
        help="强制重建 KB embedding 缓存（忽略已有缓存文件）",
    )
    parser.add_argument(
        "--save-instruction-pairs",
        action="store_true",
        help="保存原始指令与增强后指令对照文件（JSONL）",
    )
    parser.add_argument(
        "--instruction-pairs-path",
        type=Path,
        default=None,
        help="指令对照输出路径（默认写到当前 run 输出目录）",
    )
    parser.add_argument("--save-video", action="store_true", help="是否启用 internnav 保存视频（需要额外依赖）")
    parser.add_argument(
        "--no-export-images",
        action="store_true",
        help="不导出视角图到 run 目录（与 eval_retriever.py --no-export-images 语义一致；默认会导出）",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=Path("data/vln_ce/dataset_gt.csv"),
        help="GT 表路径（导出开启时用于 gt_start_view / gt_end_view；缺失则仅跳过这两项）",
    )
    args = parser.parse_args()
    if args.no_robot_image and args.robot_image is not None:
        raise SystemExit("不能同时使用 --robot-image 与 --no-robot-image")

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    _setup_sys_path(repo_root)
    print(f"[RAG4VLN] repo_root={repo_root}", flush=True)
    if args.no_robot_image:
        print(
            "[RAG4VLN] --no-robot-image：检索不调 VLM caption；若有 ins_start_view 导出仍尝试拷贝磁盘 PNG（若存在）",
            flush=True,
        )

    eval_cfg_py = Path(args.config)
    if not eval_cfg_py.is_absolute():
        eval_cfg_py = (repo_root / eval_cfg_py).resolve()

    base_eval_cfg = _load_eval_cfg_py(eval_cfg_py)
    habitat_cfg_path = base_eval_cfg.env.env_settings.get("config_path")
    if not isinstance(habitat_cfg_path, str) or not habitat_cfg_path:
        raise RuntimeError("eval cfg env.env_settings.config_path not found")
    habitat_yaml_path = Path(habitat_cfg_path)
    if not habitat_yaml_path.is_absolute():
        habitat_yaml_path = (repo_root / habitat_yaml_path).resolve()

    data_path_pattern, split, _scenes_dir = _find_data_path_from_habitat_yaml(habitat_yaml_path)
    dataset_gz_path = _resolve_dataset_gz(repo_root, data_path_pattern, split)
    if not dataset_gz_path.is_file():
        raise SystemExit(f"dataset json.gz not found: {dataset_gz_path}")
    print(f"[RAG4VLN] dataset={dataset_gz_path}", flush=True)

    # --------- rag4vln components ---------
    from src.kb import KnowledgeBase  # noqa: E402
    from src.retrieval import BinaryRandomEmbedder, Retriever  # noqa: E402
    from src.augment import retrieval_evidence_from_plan  # noqa: E402
    from src.augment import AugmentationResult  # noqa: E402
    from src.retrieval import ViTEmbedder, build_text_embedder_from_config  # noqa: E402

    rag4vln_cfg_path = (
        args.rag4vln_config if args.rag4vln_config.is_absolute() else (repo_root / args.rag4vln_config).resolve()
    )
    kb_root = args.kb_root if args.kb_root.is_absolute() else (repo_root / args.kb_root).resolve()
    kb_embed_cache_path = None
    if args.kb_embed_cache is not None:
        kb_embed_cache_path = args.kb_embed_cache if args.kb_embed_cache.is_absolute() else (repo_root / args.kb_embed_cache).resolve()

    need_config = args.text_embedder != "binary" or args.vision_embedder == "vit"
    if need_config and not rag4vln_cfg_path.is_file():
        raise SystemExit(f"rag4vln config not found: {rag4vln_cfg_path}")

    if args.text_embedder == "binary" and args.vision_embedder == "binary":
        binary_dim = max(1, int(args.binary_dim))
    elif args.text_embedder == "binary" or args.vision_embedder == "binary":
        binary_dim = _cfg_embedding_dim(rag4vln_cfg_path)
    else:
        binary_dim = 768

    print(f"[RAG4VLN] loading KB + embedders (first run / cold cache can take minutes)...", flush=True)
    kb = KnowledgeBase(kb_root)

    if args.text_embedder == "binary":
        text_e = BinaryRandomEmbedder(dim=binary_dim, threshold=0.3)
    else:
        backend = None if args.text_embedder == "auto" else args.text_embedder
        text_e = build_text_embedder_from_config(rag4vln_cfg_path, backend=backend)

    if args.vision_embedder == "binary":
        vision_e = BinaryRandomEmbedder(dim=binary_dim, threshold=0.3)
    else:
        vision_e = ViTEmbedder(config_path=rag4vln_cfg_path)

    retriever = Retriever(
        text_embedder=text_e,
        vision_embedder=vision_e,
        caption_config_path=rag4vln_cfg_path,
    )
    augmenter = _build_augmenter(args.augmenter, rag4vln_cfg_path)
    print(
        f"[RAG4VLN] retriever ready (text={args.text_embedder}, vision={args.vision_embedder}, "
        f"augmenter={args.augmenter})",
        flush=True,
    )

    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        raise SystemExit("eval_rag4vln_vln_augmented.py requires Pillow to load start_view images (pip install pillow)")

    fixed_robot_image = None
    fixed_robot_image_path: Optional[Path] = None
    if args.robot_image is not None:
        img_path = args.robot_image if args.robot_image.is_absolute() else (repo_root / args.robot_image).resolve()
        if not img_path.is_file():
            raise SystemExit(f"robot-image not found: {img_path}")
        fixed_robot_image_path = img_path.resolve()
        fixed_robot_image = _load_robot_image(img_path)
        if fixed_robot_image is None:
            raise SystemExit(f"failed to load robot-image: {img_path}")

    # --------- load dataset and augment ---------
    with gzip.open(dataset_gz_path, "rt", encoding="utf-8") as f:
        dataset_txt = json.load(f)

    episodes: List[Dict[str, Any]] = dataset_txt.get("episodes") or []
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("dataset episodes empty/invalid")

    n_all = len(episodes)
    max_episodes = int(args.max_episodes)
    if max_episodes > 0:
        episodes = list(episodes[:max_episodes])
    print(
        f"[RAG4VLN] episodes in split={n_all}, will augment+eval "
        f"{'all' if max_episodes <= 0 else f'first {len(episodes)}'} "
        f"(max_episodes={max_episodes})",
        flush=True,
    )
    start_view_root = (repo_root / "data" / "vln_ce" / "start_view").resolve()

    export_images = not bool(args.no_export_images)

    gt_rows_by_ep: Dict[str, List[Dict[str, str]]] = {}
    if export_images:
        gt_csv_path = args.gt_csv if args.gt_csv.is_absolute() else (repo_root / args.gt_csv).resolve()
        if gt_csv_path.is_file():
            gt_rows_by_ep = _load_gt_rows_by_episode(gt_csv_path)
            n_rows = sum(len(v) for v in gt_rows_by_ep.values())
            n_dup = sum(1 for v in gt_rows_by_ep.values() if len(v) > 1)
            print(
                f"[RAG4VLN] gt_csv loaded episodes={len(gt_rows_by_ep)} rows={n_rows} "
                f"duplicate_episode_id={n_dup} path={gt_csv_path}",
                flush=True,
            )
        else:
            print(
                f"[export] gt csv not found, skip gt_start_view/gt_end_view only: {gt_csv_path}",
                flush=True,
            )

    saved_at = datetime.datetime.now().astimezone()
    ts = saved_at.strftime("%Y%m%d_%H%M%S")
    out_dir = (repo_root / "rag4vln/results/augmented_vln_eval").resolve() / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RAG4VLN] run output dir={out_dir}", flush=True)

    out_episodes = episodes
    instruction_pairs: List[Dict[str, Any]] = []
    n_ep = len(out_episodes)
    print(f"[RAG4VLN] --- augmentation phase ({n_ep} rows) ---", flush=True)
    pbar = tqdm(
        total=n_ep,
        desc="RAG4VLN augment",
        unit="ep",
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=0.2,
    )
    for idx, ep in enumerate(out_episodes):
        pbar.set_postfix(idx=idx, phase="prepare", refresh=True)
        instr = (ep.get("instruction") or {}).get("instruction_text")
        if not isinstance(instr, str) or not instr.strip():
            print(f"[RAG4VLN] ep {idx+1}/{n_ep} skip (empty instruction)", flush=True)
            pbar.update(1)
            continue
        original_instr = instr.strip()
        _prev = original_instr[:80] + ("..." if len(original_instr) > 80 else "")
        print(f"[RAG4VLN] ep {idx+1}/{n_ep} instruction={_prev!r}", flush=True)
        t0 = time.perf_counter()
        ep_id = str(ep.get("episode_id", "")).strip()
        start_pos = ep.get("start_position")
        robot_position = start_pos if isinstance(start_pos, list) and len(start_pos) == 3 else [0.0, 0.0, 0.0]
        ep_img_path: Optional[Path] = None
        robot_image: Any = None
        if args.no_robot_image:
            if ep_id:
                cand = _resolve_start_view_png(repo_root, start_view_root, ep_id, dataset_gz_path)
                ep_img_path = cand if cand.is_file() else None
            robot_image = None
        elif fixed_robot_image is not None:
            robot_image = fixed_robot_image
            ep_img_path = fixed_robot_image_path
        else:
            if not ep_id:
                raise SystemExit("episode_id missing for start_view resolution")
            ep_img_path = _resolve_start_view_png(repo_root, start_view_root, ep_id, dataset_gz_path)
            if not ep_img_path.is_file():
                raise SystemExit(
                    f"missing start_view image for episode_id={ep_id}: {ep_img_path} "
                    f"(run rag4vln/scripts/build_dataset_gt.py or fix start_view path)"
                )
            robot_image = _load_robot_image(ep_img_path)
            if robot_image is None:
                raise SystemExit(f"failed to load start_view image for episode_id={ep_id}: {ep_img_path}")
        pbar.set_postfix(idx=idx, phase="retrieve", refresh=True)
        print(f"[RAG4VLN] ep {idx+1}/{n_ep} phase=retrieve ...", flush=True)
        plan = retriever.retrieve(
            kb,
            instruction=original_instr,
            robot_position=robot_position,  # type: ignore[arg-type]
            robot_image=robot_image,
            topk1_scenes=args.topk1,
            topk2_zones=args.topk2,
            topk3_pairs=args.topk3,
            embed_view_images=(args.vision_embedder == "vit"),
            timing=False,
            verbose=False,
            progress=False,
            kb_cache_path=kb_embed_cache_path,
            force_rebuild_kb_cache=bool(args.rebuild_kb_embed_cache),
        )
        n_pairs = len(plan.get("topk3_pairs") or [])
        print(
            f"[RAG4VLN] ep {idx+1}/{n_ep} retrieve done (topk3_pairs={n_pairs})",
            flush=True,
        )
        if export_images and ep_id:
            _export_ins_start_view(out_dir, ep_id, ep_img_path)
            _export_retriever_rank_view_images(kb, plan, out_dir, ep_id, int(args.topk3))
            gt_row = _lookup_gt_row(gt_rows_by_ep, ep_id, ep.get("scene_id"))
            if gt_row:
                _export_gt_kb_endpoint_views(
                    kb,
                    out_dir,
                    ep_id,
                    gt_row["gt_scene_id"],
                    gt_row["gt_start_view_id"],
                    gt_row["gt_end_view_id"],
                )
            elif gt_rows_by_ep:
                rows = gt_rows_by_ep.get(ep_id) or []
                if len(rows) > 1:
                    print(
                        f"[export] ambiguous GT for episode_id={ep_id} "
                        f"episode_scene={_parse_kb_scene_id(ep.get('scene_id'))!r} "
                        f"csv_scenes={[r['gt_scene_id'] for r in rows]}, skip gt_start/gt_end",
                        flush=True,
                    )
                else:
                    print(f"[export] no GT row for episode_id={ep_id}, skip gt_start/gt_end", flush=True)
        elif export_images and not ep_id:
            print(f"[export] ep index={idx} has empty episode_id, skip view export", flush=True)

        evidence = retrieval_evidence_from_plan(plan)
        path_regions = _extract_path_region_descriptions(kb, plan)
        pbar.set_postfix(idx=idx, phase="augment", refresh=True)
        print(f"[RAG4VLN] ep {idx+1}/{n_ep} phase=augment (LLM) ...", flush=True)
        try:
            aug_res = augmenter.augment(original_instr, evidence, path_region_descriptions=path_regions)
        except Exception as e:
            print(
                f"[RAG4VLN] ep {idx+1}/{n_ep} augment EXCEPTION {type(e).__name__}: {e}; "
                f"use original instruction",
                flush=True,
            )
            aug_res = AugmentationResult(
                instruction=original_instr,
                raw_model_output=None,
                meta={
                    "fallback": True,
                    "reason": "augment_exception",
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                },
            )
        new_instr = aug_res.instruction.strip()
        meta = getattr(aug_res, "meta", None) or {}
        if meta.get("fallback"):
            print(
                f"[RAG4VLN] ep {idx+1}/{n_ep} augment FALLBACK reason={meta.get('reason')!r}",
                flush=True,
            )
        dt = time.perf_counter() - t0
        pbar.set_postfix(idx=idx, phase="done", sec=f"{dt:.1f}", refresh=True)
        _outp = new_instr[:80] + ("..." if len(new_instr) > 80 else "")
        print(
            f"[RAG4VLN] ep {idx+1}/{n_ep} augment done in {dt:.1f}s, augmented={_outp!r}",
            flush=True,
        )
        ep.setdefault("instruction", {})
        ep["instruction"]["instruction_text"] = new_instr
        if args.save_instruction_pairs:
            instruction_pairs.append(
                {
                    "idx": idx,
                    "episode_id": ep.get("episode_id"),
                    "scene_id": ep.get("scene_id"),
                    "augmenter": args.augmenter,
                    "original_instruction_text": original_instr,
                    "augmented_instruction_text": new_instr,
                    # 记录大模型原始回复，便于回看 SPP 三阶段输出与排查解析问题。
                    "raw_model_output": getattr(aug_res, "raw_model_output", None),
                    # 记录结构化中间信息（如 task_narrative / waypoints / fallback 原因等）。
                    "augment_meta": meta,
                }
            )
        # tokens 不强制重算；目前 evaluator prompt 主要依赖 instruction_text
        pbar.update(1)
    pbar.close()
    print("[RAG4VLN] augmentation phase finished", flush=True)

    # --------- write augmented json.gz + temp habitat yaml + temp eval cfg ---------
    dataset_out_gz = out_dir / f"{dataset_gz_path.stem}_aug_{args.augmenter}_{ts}.json.gz"
    dataset_txt["episodes"] = out_episodes
    with gzip.open(dataset_out_gz, "wt", encoding="utf-8") as f:
        f.write(json.dumps(dataset_txt, ensure_ascii=False))

    if args.save_instruction_pairs:
        pairs_path = args.instruction_pairs_path
        if pairs_path is None:
            pairs_path = out_dir / "instruction_pairs.jsonl"
        elif not pairs_path.is_absolute():
            pairs_path = (repo_root / pairs_path).resolve()
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        with pairs_path.open("w", encoding="utf-8") as f:
            for row in instruction_pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[augment] instruction pairs saved: {pairs_path}")

    habitat_yaml_out = out_dir / f"vln_r2r_aug_{ts}.yaml"
    # 复用 habitat yaml，仅替换 data_path
    habitat_yaml_text = habitat_yaml_path.read_text(encoding="utf-8")
    pattern = r"(^\s*data_path:\s*).*$"
    repl = r"\1" + json.dumps(str(dataset_out_gz), ensure_ascii=False)
    habitat_yaml_text2, n = re.subn(pattern, repl, habitat_yaml_text, flags=re.MULTILINE, count=1)
    if n != 1:
        raise RuntimeError(f"Failed to patch habitat yaml data_path: {habitat_yaml_out}")
    habitat_yaml_out.write_text(habitat_yaml_text2, encoding="utf-8")

    eval_cfg_out_py = out_dir / f"habitat_dual_system_cfg_aug_{ts}.py"
    # 替换 config_path 为临时 habitat yaml
    text = eval_cfg_py.read_text(encoding="utf-8")
    text2, n = re.subn(
        r"('config_path'\s*:\s*')([^']+)(')",
        lambda m: m.group(1) + str(habitat_yaml_out) + m.group(3),
        text,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"Failed to patch eval cfg config_path: {eval_cfg_out_py}")
    eval_cfg_out_py.write_text(text2, encoding="utf-8")

    # 强制输出路径唯一，避免 progress.json 造成 0it
    output_path_unique = str((out_dir / "internnav_output").resolve())
    eval_cfg_final_py = _patch_eval_cfg_output_path(eval_cfg_out_py, output_path_unique)

    # 如果不保存视频，强制 save_video False（避免 cv2 依赖）
    if not args.save_video:
        etxt = eval_cfg_final_py.read_text(encoding="utf-8")
        etxt2, _n = re.subn(r'("save_video"\s*:\s*)True\b', r'\1False', etxt, flags=re.IGNORECASE)
        eval_cfg_final_py.write_text(etxt2, encoding="utf-8")

    # --------- run internnav eval in current process (recommended) ---------
    from internnav.evaluator import Evaluator  # noqa: E402

    print(f"[RAG4VLN] --- Habitat VLN eval (internnav) --- cfg={eval_cfg_final_py}", flush=True)
    run_cfg = _load_eval_cfg_py(eval_cfg_final_py)
    evaluator = Evaluator.init(run_cfg)
    evaluator.eval()
    print("[RAG4VLN] eval finished", flush=True)


if __name__ == "__main__":
    main()

