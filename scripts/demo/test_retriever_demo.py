#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieval demo: KB + ``Retriever.retrieve``; print/save plan (no instruction augmentation).

Run from repo root: ``python rag4vln/scripts/demo/test_retriever_demo.py``
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Optional

RAG4VLN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAG4VLN_CONFIG = RAG4VLN_ROOT / "src" / "config.yaml"
DEFAULT_KB = RAG4VLN_ROOT / "data" / "kb" / "memory"
DEFAULT_ROBOT_IMAGE = RAG4VLN_ROOT / "data" / "test_materials" / "test.png"


def _setup_sys_path() -> None:
    if str(RAG4VLN_ROOT) not in sys.path:
        sys.path.insert(0, str(RAG4VLN_ROOT))


_setup_sys_path()

from src.kb import KnowledgeBase  # noqa: E402
from src.retrieval import BinaryRandomEmbedder, Retriever  # noqa: E402


def _load_robot_image(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"Robot test image not found: {path.resolve()}")
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise ImportError("Loading images requires pillow: pip install pillow") from e
    return Image.open(path).convert("RGB")


def _cfg_embedding_dim(config_path: Path) -> int:
    from src.config_io import retrieval_embedding_dim  # noqa: E402

    try:
        return retrieval_embedding_dim(config_path)
    except Exception:
        return 768


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


def main() -> None:
    parser = argparse.ArgumentParser(description="rag4vln Retriever-only demo")
    parser.add_argument("--text-embedder", choices=("auto", "bert", "sbert", "bge", "binary"), default="auto")
    parser.add_argument("--vision-embedder", choices=("vit", "binary"), default="vit")
    parser.add_argument("--config", type=Path, default=None, help=f"rag4vln unified config (default {DEFAULT_RAG4VLN_CONFIG})")
    parser.add_argument("--kb-root", type=Path, default=DEFAULT_KB)
    parser.add_argument("--instruction", type=str, default="Represent this sentence for searching relevant passages: I want to watch TV")
    parser.add_argument("--binary-dim", type=int, default=64)
    parser.add_argument("--robot-image", type=Path, default=None, help=f"Robot observation image (default {DEFAULT_ROBOT_IMAGE})")
    parser.add_argument("--no-robot-image", action="store_true", help="No image; skip VLM caption (legacy default)")
    parser.add_argument(
        "--retrieve-verbose",
        action="store_true",
        help="Print Retriever debug logs (incl. instruction string passed to text_embedder, for BGE prefix check)",
    )
    parser.add_argument("--result-dir", type=Path, default=RAG4VLN_ROOT / "results")
    parser.add_argument("--no-save-result", action="store_true")
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

    kb_root = args.kb_root if args.kb_root.is_absolute() else RAG4VLN_ROOT / args.kb_root
    kb = KnowledgeBase(kb_root)
    text_e = _build_text_embedder(args.text_embedder, cfg, binary_dim=binary_dim)
    vision_e = _build_vision_embedder(args.vision_embedder, cfg, binary_dim=binary_dim)
    retriever = Retriever(
        text_embedder=text_e,
        vision_embedder=vision_e,
        caption_config_path=cfg,
    )

    if args.no_robot_image:
        robot_image = None
    else:
        robot_path = args.robot_image if args.robot_image is not None else DEFAULT_ROBOT_IMAGE
        if not robot_path.is_absolute():
            robot_path = Path.cwd() / robot_path
        robot_image = _load_robot_image(robot_path)

    instruction = args.instruction.strip()
    plan = retriever.retrieve(
        kb,
        instruction=instruction,
        robot_position=[0.0, 0.0, 0.0],
        robot_image=robot_image,
        topk1_scenes=3,
        topk2_zones=3,
        topk3_pairs=3,
        embed_view_images=(args.vision_embedder == "vit"),
        timing=True,
        verbose=bool(args.retrieve_verbose),
        progress=True,
    )

    print(f"[demo] text={args.text_embedder} vision={args.vision_embedder}")
    print(json.dumps({k: plan[k] for k in plan if k != "timing_ms"}, ensure_ascii=False, indent=2, default=str))
    if "timing_ms" in plan:
        print("\n=== timing_ms ===")
        print(json.dumps(plan["timing_ms"], ensure_ascii=False, indent=2))

    if args.no_save_result:
        return

    saved_at = datetime.datetime.now().astimezone()
    ts = saved_at.strftime("retrieve_%Y%m%d_%H%M%S")
    run_dir = args.result_dir.expanduser().resolve() / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "result.txt").write_text(
        f"saved_at: {saved_at.isoformat(timespec='seconds')}\ninstruction: {instruction}\n",
        encoding="utf-8",
    )
    print(f"\n[saved] {run_dir}")


if __name__ == "__main__":
    main()
