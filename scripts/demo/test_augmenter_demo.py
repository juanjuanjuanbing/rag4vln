#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retriever + Augmenter demo (switchable augmenters).

Augmenter choices:
- llm_direct
- template_path
- semantic_pathplanning (single LLM: evidence CoT → semantic waypoints → FSM/VLN phrasing)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _setup_sys_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    rag4vln_root = repo_root / "rag4vln"
    if str(rag4vln_root) not in sys.path:
        sys.path.insert(0, str(rag4vln_root))


_setup_sys_path()

from src.augment import (  # noqa: E402
    build_llm_direct_augmenter,
    build_semantic_pathplanning_augmenter,
    build_template_path_augmenter,
    retrieval_evidence_from_plan,
)
from src.augment.types import AugmentationResult  # noqa: E402
from src.kb import KnowledgeBase  # noqa: E402
from src.retrieval import BinaryRandomEmbedder, Retriever  # noqa: E402


RAG4VLN_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_RAG4VLN_CONFIG = RAG4VLN_ROOT / "src" / "config.yaml"
DEFAULT_ROBOT_IMAGE = RAG4VLN_ROOT / "data" / "test_materials" / "test.png"
DEFAULT_START_VIEW_IMAGE = RAG4VLN_ROOT / "data" / "vln_ce" / "start_view" / "data" / "vln_ce" / "raw_data" / "r2r" / "val_seen" / "ep_1.png"
DEFAULT_GT_CSV = RAG4VLN_ROOT.parent / "data" / "vln_ce" / "dataset_gt.csv"


def _cfg_embedding_dim(config_path: Path) -> int:
    from src.config_io import retrieval_embedding_dim  # noqa: E402

    try:
        return retrieval_embedding_dim(config_path)
    except Exception:
        return 768


def _load_robot_image(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"Robot test image not found: {path.resolve()}")
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise ImportError("Loading images requires pillow: pip install pillow") from e
    return Image.open(path).convert("RGB")


def _build_text_embedder(kind: str, config_path: Optional[Path], *, binary_dim: int) -> Any:
    if kind == "binary":
        return BinaryRandomEmbedder(dim=binary_dim, threshold=0.3)
    from src.retrieval import build_text_embedder_from_config  # noqa: E402

    backend = None if kind == "auto" else kind
    return build_text_embedder_from_config(config_path, backend=backend)


def _build_vision_embedder(kind: str, config_path: Optional[Path], *, binary_dim: int) -> Any:
    if kind == "binary":
        return BinaryRandomEmbedder(dim=binary_dim, threshold=0.3)
    from src.retrieval import ViTEmbedder  # noqa: E402

    return ViTEmbedder(config_path=config_path)


def _save_kb_view_image(kb: KnowledgeBase, scene_id: str, view_id: str, dest: Path) -> Tuple[bool, str]:
    p = kb.view_image_path(scene_id, view_id)
    if p is not None and p.is_file():
        shutil.copy2(p, dest)
        return True, str(p)
    img = kb.load_view_image(scene_id, view_id)
    if img is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return True, "(pil)"
    return False, ""


def _extract_path_region_descriptions(kb: KnowledgeBase, plan: dict) -> list[str]:
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
    out: list[str] = []
    seen: set[str] = set()
    for zid in pair.get("path_zone_ids") or []:
        if not isinstance(zid, str) or not zid or zid in seen:
            continue
        seen.add(zid)
        zdesc = ((zones.get(zid) or {}).get("attributes") or {}).get("description")
        s = str(zdesc).strip() if zdesc is not None else ""
        out.append(s if s else zid)
    return out


def _build_augmenter(kind: str, rag4vln_config: Path):
    cfg = rag4vln_config if rag4vln_config.is_file() else None
    if kind == "template_path":
        return build_template_path_augmenter(config_path=cfg)
    if kind == "semantic_pathplanning":
        return build_semantic_pathplanning_augmenter(config_path=cfg)
    return build_llm_direct_augmenter(config_path=cfg)


def _augmentation_record(original: str, aug: AugmentationResult) -> dict[str, Any]:
    return {
        "original_instruction": original,
        "augmented_instruction": aug.instruction,
        "raw_model_output": aug.raw_model_output,
        "meta": dict(aug.meta) if aug.meta else {},
    }


def _lookup_episode_row(gt_csv: Path, episode_id: str) -> Optional[dict[str, str]]:
    if not gt_csv.is_file():
        return None
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("episode_id", "")).strip() == str(episode_id).strip():
                return {k: str(v or "").strip() for k, v in row.items()}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Retriever + Instruction Augmenter comparison demo")
    parser.add_argument(
        "--augmenter",
        choices=("llm_direct", "template_path", "semantic_pathplanning"),
        default="llm_direct",
    )
    parser.add_argument("--text-embedder", choices=("auto", "bert", "sbert", "bge", "binary"), default="auto")
    parser.add_argument("--vision-embedder", choices=("vit", "binary"), default="vit")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"rag4vln unified config (retrieval / augment); default {DEFAULT_RAG4VLN_CONFIG}",
    )
    parser.add_argument(
        "--start-view-image",
        type=Path,
        default=None,
        help=f"Start-view image (overrides --robot-image), e.g. {DEFAULT_START_VIEW_IMAGE}",
    )
    parser.add_argument("--episode-id", type=str, default=None, help="Load start image and instruction from GT CSV by episode_id")
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV, help="GT CSV with start_view_image_path")
    parser.add_argument("--robot-image", type=Path, default=None, help=f"Robot image (default {DEFAULT_ROBOT_IMAGE})")
    parser.add_argument("--instruction", type=str, default="Represent this sentence for searching relevant passages: I want to watch TV")
    parser.add_argument("--binary-dim", type=int, default=64)
    parser.add_argument("--result-dir", type=Path, default=RAG4VLN_ROOT / "results")
    parser.add_argument("--no-save-result", action="store_true")
    parser.add_argument(
        "--retrieve-verbose",
        action="store_true",
        help="Print Retriever debug logs (incl. instruction string passed to text_embedder)",
    )
    args = parser.parse_args()

    cfg = args.config if args.config is not None else DEFAULT_RAG4VLN_CONFIG
    need_config = args.text_embedder != "binary" or args.vision_embedder == "vit"
    if need_config and not cfg.is_file():
        raise SystemExit(f"Retrieval config not found: {cfg}")

    if args.text_embedder == "binary" and args.vision_embedder == "binary":
        binary_dim = max(1, int(args.binary_dim))
    elif args.text_embedder == "binary" or args.vision_embedder == "binary":
        binary_dim = _cfg_embedding_dim(cfg)
    else:
        binary_dim = 768

    kb = KnowledgeBase(RAG4VLN_ROOT / "data" / "kb" / "memory")
    text_e = _build_text_embedder(args.text_embedder, cfg, binary_dim=binary_dim)
    vision_e = _build_vision_embedder(args.vision_embedder, cfg, binary_dim=binary_dim)
    retriever = Retriever(
        text_embedder=text_e,
        vision_embedder=vision_e,
        caption_config_path=cfg,
    )

    instruction = args.instruction.strip()
    robot_path: Path
    if args.episode_id is not None:
        gt_csv = args.gt_csv if args.gt_csv.is_absolute() else (RAG4VLN_ROOT.parent / args.gt_csv).resolve()
        row = _lookup_episode_row(gt_csv, args.episode_id)
        if row is None:
            raise SystemExit(f"episode_id={args.episode_id} not found in GT CSV: {gt_csv}")
        if row.get("instruction_text"):
            instruction = row["instruction_text"].strip()
        rel = row.get("start_view_image_path", "").strip()
        if not rel:
            raise SystemExit(f"episode_id={args.episode_id} missing start_view_image_path")
        rp = Path(rel)
        robot_path = rp if rp.is_absolute() else (RAG4VLN_ROOT.parent / rp).resolve()
    elif args.start_view_image is not None:
        robot_path = args.start_view_image if args.start_view_image.is_absolute() else (Path.cwd() / args.start_view_image).resolve()
    else:
        rp = args.robot_image if args.robot_image is not None else DEFAULT_ROBOT_IMAGE
        robot_path = rp if rp.is_absolute() else (Path.cwd() / rp).resolve()
    robot_image = _load_robot_image(robot_path)
    robot_position = [0.0, 0.0, 0.0]

    plan = retriever.retrieve(
        kb,
        instruction=instruction,
        robot_position=robot_position,
        robot_image=robot_image,
        topk1_scenes=3,
        topk2_zones=3,
        topk3_pairs=3,
        embed_view_images=(args.vision_embedder == "vit"),
        timing=True,
        verbose=bool(args.retrieve_verbose),
        progress=True,
    )

    evidence = retrieval_evidence_from_plan(plan)
    path_regions = _extract_path_region_descriptions(kb, plan)
    augmenter = _build_augmenter(args.augmenter, cfg)
    aug_res = augmenter.augment(instruction, evidence, path_region_descriptions=path_regions)

    print(f"[demo] augmenter={args.augmenter} text={args.text_embedder} vision={args.vision_embedder}")
    print("\n=== Enhanced Instruction ===")
    print(aug_res.instruction)
    if aug_res.meta:
        print("\n=== Augmentation Meta ===")
        print(json.dumps(aug_res.meta, ensure_ascii=False, indent=2, default=str))

    if args.no_save_result:
        return

    saved_at = datetime.datetime.now().astimezone()
    ts = saved_at.strftime("augment_%Y%m%d_%H%M%S")
    run_dir = args.result_dir.expanduser().resolve() / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "augmentation.json").write_text(
        json.dumps(_augmentation_record(instruction, aug_res), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (run_dir / "result.txt").write_text(
        f"saved_at: {saved_at.isoformat(timespec='seconds')}\n"
        f"augmenter: {args.augmenter}\n"
        f"instruction: {instruction}\n"
        f"augmented_instruction: {aug_res.instruction}\n",
        encoding="utf-8",
    )

    pairs = plan.get("topk3_pairs") or []
    if pairs and isinstance(pairs[0], dict):
        p0 = pairs[0]
        sid = p0.get("scene_id")
        sv = p0.get("start_view_id")
        ev = p0.get("end_view_id")
        if isinstance(sid, str) and isinstance(sv, str):
            _save_kb_view_image(kb, sid, sv, run_dir / "start_view.png")
        if isinstance(sid, str) and isinstance(ev, str):
            _save_kb_view_image(kb, sid, ev, run_dir / "goal_view.png")

    print(f"\n[saved] {run_dir}")


if __name__ == "__main__":
    main()
