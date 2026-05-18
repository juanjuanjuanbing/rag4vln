#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 KB 中某场景的 view 图片及同名 JSON（view 节点 ``attributes`` 等结构化内容）。

在仓库根目录执行：``python rag4vln/scripts/demo/visualize_kb_views.py``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

RAG4VLN_ROOT = Path(__file__).resolve().parents[2]


def _setup_sys_path() -> None:
    if str(RAG4VLN_ROOT) not in sys.path:
        sys.path.insert(0, str(RAG4VLN_ROOT))


_setup_sys_path()

from src.kb import KnowledgeBase  # noqa: E402


def _view_record(vnode: Dict[str, Any]) -> Dict[str, Any]:
    attrs = dict(vnode.get("attributes") or {})
    out: Dict[str, Any] = {"attributes": attrs}
    for k in ("id", "type", "label"):
        if k in vnode:
            out[k] = vnode[k]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export KB view PNG + sidecar JSON")
    parser.add_argument("--scene-id", type=str, default="Pm6F8kyY3z2")
    parser.add_argument("--kb-root", type=Path, default=RAG4VLN_ROOT / "data" / "kb" / "memory")
    parser.add_argument("--out-dir", type=Path, default=RAG4VLN_ROOT / "results" / "kb_vis")
    parser.add_argument("--all-views", action="store_true", help="导出该场景全部 view")
    parser.add_argument("--max-views", type=int, default=8, help="未指定 --all-views 时最多导出数量")
    args = parser.parse_args()

    kb_root = args.kb_root if args.kb_root.is_absolute() else RAG4VLN_ROOT / args.kb_root
    kb = KnowledgeBase(kb_root)
    tree = kb.scene(args.scene_id)
    views = tree.get("views") or {}
    if not isinstance(views, dict) or not views:
        raise SystemExit(f"场景 {args.scene_id!r} 无 views")

    out_dir = args.out_dir.expanduser().resolve() / args.scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = list(views.keys())
    if not args.all_views:
        vids = vids[: max(1, int(args.max_views))]

    n_img = 0
    for vid in vids:
        vnode = views.get(vid)
        if not isinstance(vnode, dict):
            continue
        base = out_dir / f"kb_viz_{args.scene_id}_{vid}"
        json_path = base.with_suffix(".json")
        json_path.write_text(
            json.dumps(_view_record(vnode), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        src = kb.view_image_path(args.scene_id, str(vid))
        if src is not None and src.is_file():
            shutil.copy2(src, base.with_suffix(".png"))
            n_img += 1
        else:
            img = kb.load_view_image(args.scene_id, str(vid))
            if img is not None:
                img.save(base.with_suffix(".png"))
                n_img += 1

    print(
        f"[visualize_kb_views] scene={args.scene_id} json={len(vids)} png_ok={n_img} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
