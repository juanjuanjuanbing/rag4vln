#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build memory KB (logic in ``src.kb.build_knowledgebase_from_memory``).
Optional ``--render-view-images`` → ``attach_view_images_to_kb`` (Habitat: ``src.utils.habitat_render``).
"""

import argparse
import sys
from pathlib import Path


def _setup_sys_path() -> None:
    root = Path(__file__).resolve().parents[2]
    sr = root / "rag4vln"
    if str(sr) not in sys.path:
        sys.path.insert(0, str(sr))


_setup_sys_path()

from src.kb import attach_view_images_to_kb, build_knowledgebase_from_memory


def main() -> None:
    p = argparse.ArgumentParser(description="Build memory KB.")
    p.add_argument("--memory-dir", default="rag4vln/data/memory/instruction_generator")
    p.add_argument("--output-dir", default="rag4vln/data/kb/memory")
    p.add_argument("--connectivity-subdirs", nargs="*", default=["connectivity_mp3d"])
    p.add_argument(
        "--view-annotation",
        default="rag4vln/data/memory/instruction_generator/mp3d_view_annotation.json",
    )
    p.add_argument(
        "--zone-annotation",
        default="rag4vln/data/memory/instruction_generator/mp3d_zone_annotation.json",
    )
    p.add_argument(
        "--house-annotation",
        default="rag4vln/data/memory/instruction_generator/mp3d_house_annotation.json",
    )
    p.add_argument("--render-view-images", action="store_true")
    p.add_argument("--scene-root", default="data/scene_data/mp3d_ce/mp3d")
    p.add_argument("--render-width", type=int, default=640)
    p.add_argument("--render-height", type=int, default=480)
    p.add_argument("--render-hfov", type=float, default=90.0)
    args = p.parse_args()

    pr = Path(__file__).resolve().parents[2]
    mem = pr / args.memory_dir
    out = pr / args.output_dir

    kb = build_knowledgebase_from_memory(
        mem,
        out,
        connectivity_subdirs=args.connectivity_subdirs,
        house_annotation_path=pr / args.house_annotation
        if (pr / args.house_annotation).exists()
        else None,
        view_annotation_path=pr / args.view_annotation
        if (pr / args.view_annotation).exists()
        else None,
        zone_annotation_path=pr / args.zone_annotation
        if (pr / args.zone_annotation).exists()
        else None,
    )
    nsc = len(kb.list_scene_ids())
    print(f"KB done: {out}, scenes={nsc}, retrieval_docs={kb.num_documents()}")

    if args.render_view_images:
        ns, nk = attach_view_images_to_kb(
            kb,
            pr / args.scene_root,
            width=args.render_width,
            height=args.render_height,
            hfov=args.render_hfov,
        )
        print(f"View images: saved {ns}, skipped {nk}.")


if __name__ == "__main__":
    main()
