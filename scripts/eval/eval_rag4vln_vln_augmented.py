# -*- coding: utf-8 -*-
"""
rag4vln eval entry plugged into InternVLA (HabitatVLN).

Goals:
- Mirror ``internnav/scripts/eval/eval.py`` minimal CLI: ``--config <eval_cfg.py>``
- Before eval: run rag4vln retrieval + instruction augmentation on each episode's
  ``instruction.instruction_text``, replace with augmented text, then run Habitat eval.

Default (reproducibility + avoid progress.json skip):
- Patch eval cfg in place so ``eval_settings.output_path`` points to a fresh run directory.

View export matches ``eval_retriever.py``: default on under this run dir; ``--no-export-images`` disables.
Writes ``ins_start_view/``, ``retriever_start_view/``, ``retriever_end_view/``; with ``--gt-csv`` also ``gt_start_view/``, ``gt_end_view/``.

Tip: with ``conda run``, add ``--no-capture-output`` if logs appear only at exit (buffered). See ``rag4vln/scripts/eval/README.md``.
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
    # enable `import internnav.*`
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
    Patch eval_settings.output_path in place on the eval cfg .py (no extra long-filename copy).
    """
    text = eval_cfg_py.read_text(encoding="utf-8")
    # replace only the first output_path: "..."
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
    """Shared ``start_view/r2r/<split>/ep_<id>.png``; same as eval_retriever / build_dataset_gt from_r2r."""
    vln_ce_root = (repo_root / "data" / "vln_ce").resolve()
    ds_parent = dataset_file.expanduser().resolve().parent
    try:
        rel_parts = ds_parent.relative_to(vln_ce_root).parts
    except ValueError as e:
        raise SystemExit(
            f"start_view path requires dataset JSON under data/vln_ce: {dataset_file}"
        ) from e
    try:
        i = rel_parts.index("r2r")
    except ValueError as e:
        raise SystemExit(
            f"start_view shared path requires an r2r segment in the path relative to vln_ce: {dataset_file}"
        ) from e
    sub = Path(*rel_parts[i:])
    return (start_view_root / sub / f"ep_{ep_id}.png").resolve()


def _parse_kb_scene_id(vln_scene_id: Any) -> str:
    """Same as ``build_dataset_gt._parse_kb_scene_id``: episode.scene_id → KB scene name."""
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
    """Same ``episode_id`` may appear in val_seen and val_unseen in ``dataset_gt.csv``; keep a list per id."""
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
    """Same as ``eval_retriever.py``: copy KB file on disk if present, else save via PIL."""
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
    """Export top-k KB start/end view images from retrieval ``topk3_pairs`` (same dirs as retriever eval)."""
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
    """``ins_start_view``: start observation PNG used for retrieval (same as eval_retriever)."""
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
    """``gt_start_view`` / ``gt_end_view``: GT CSV KB panoramas (not in retriever eval; for comparison)."""
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
            "Tip: `conda run` captures stdout/stderr by default; long runs may look like no output.\n"
            "Add `--no-capture-output` (or `--live-stream`) for live logs, e.g.:\n"
            "  conda run --no-capture-output -n inter_hab python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py ...\n"
            "Line continuations with backslash must be the last character on the line; do not split `--save-instruction-pairs`.\n"
            "Offline or DNS failure (OpenAI APIConnectionError): use `--no-robot-image` to skip VLM caption during retrieval."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py",
        help="InternNav eval cfg .py (relative to repo_root or absolute)",
    )
    parser.add_argument(
        "--augmenter",
        choices=("llm_direct", "template_path", "semantic_pathplanning", "r_only"),
        default="semantic_pathplanning",
        help="Includes r_only: concat retrieval evidence + original instruction (no LLM)",
    )
    parser.add_argument(
        "--rag4vln-config",
        type=Path,
        default=Path("rag4vln/src/config.yaml"),
        help="rag4vln unified YAML (basic / retrieval / augment)",
    )
    parser.add_argument("--kb-root", type=Path, default=Path("rag4vln/data/kb/memory"))
    parser.add_argument("--text-embedder", choices=("auto", "bert", "sbert", "bge", "binary"), default="binary")
    parser.add_argument("--vision-embedder", choices=("vit", "binary"), default="binary")
    parser.add_argument("--binary-dim", type=int, default=64)
    parser.add_argument("--topk1", type=int, default=3)
    parser.add_argument("--topk2", type=int, default=3)
    parser.add_argument("--topk3", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=1, help="Augment and eval only the first N episodes")
    parser.add_argument("--robot-image", type=Optional[Path], default=None, help="Optional fixed start image for all episodes")
    parser.add_argument(
        "--no-robot-image",
        action="store_true",
        help="No start observation for retrieval: skip VLM caption (no OpenAI); zero robot image/text query emb; offline-friendly with r_only",
    )
    parser.add_argument(
        "--kb-embed-cache",
        type=Path,
        default=None,
        help="KB embedding cache file (.pt); reuse after first build for large speedup",
    )
    parser.add_argument(
        "--rebuild-kb-embed-cache",
        action="store_true",
        help="Force rebuild KB embedding cache (ignore existing file)",
    )
    parser.add_argument(
        "--save-instruction-pairs",
        action="store_true",
        help="Save original vs augmented instruction pairs (JSONL)",
    )
    parser.add_argument(
        "--instruction-pairs-path",
        type=Path,
        default=None,
        help="Path for instruction pair JSONL (default: current run output dir)",
    )
    parser.add_argument("--save-video", action="store_true", help="Enable internnav video saving (extra deps)")
    parser.add_argument(
        "--no-export-images",
        action="store_true",
        help="Do not export view images to run dir (same as eval_retriever --no-export-images; export is default)",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=Path("data/vln_ce/dataset_gt.csv"),
        help="GT CSV path for gt_start_view / gt_end_view when exporting; skip those dirs if missing",
    )
    args = parser.parse_args()
    if args.no_robot_image and args.robot_image is not None:
        raise SystemExit("Cannot use --robot-image and --no-robot-image together")

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
            "[RAG4VLN] --no-robot-image: retrieval skips VLM caption; ins_start_view export still copies on-disk PNG if present",
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
                    # Raw LLM reply for SPP three-stage review and parse debugging.
                    "raw_model_output": getattr(aug_res, "raw_model_output", None),
                    # Structured intermediates (task_narrative / waypoints / fallback reason, etc.).
                    "augment_meta": meta,
                }
            )
        # tokens not recomputed; evaluator prompt mainly uses instruction_text
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
    # reuse habitat yaml, only patch data_path
    habitat_yaml_text = habitat_yaml_path.read_text(encoding="utf-8")
    pattern = r"(^\s*data_path:\s*).*$"
    repl = r"\1" + json.dumps(str(dataset_out_gz), ensure_ascii=False)
    habitat_yaml_text2, n = re.subn(pattern, repl, habitat_yaml_text, flags=re.MULTILINE, count=1)
    if n != 1:
        raise RuntimeError(f"Failed to patch habitat yaml data_path: {habitat_yaml_out}")
    habitat_yaml_out.write_text(habitat_yaml_text2, encoding="utf-8")

    eval_cfg_out_py = out_dir / f"habitat_dual_system_cfg_aug_{ts}.py"
    # point config_path at temporary habitat yaml
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

    # unique output path to avoid progress.json skipping all episodes (0it)
    output_path_unique = str((out_dir / "internnav_output").resolve())
    eval_cfg_final_py = _patch_eval_cfg_output_path(eval_cfg_out_py, output_path_unique)

    # if not saving video, force save_video False (avoids cv2 dependency)
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

