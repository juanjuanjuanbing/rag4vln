#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 MP3D connectivity JSON（``data/memory/instruction_generator/connectivity_mp3d``），打印统计并可写摘要文件。

在仓库根目录执行：``python rag4vln/scripts/demo/visualize_connectivity_mp3d.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAG4VLN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONN_DIR = (
    RAG4VLN_ROOT / "data" / "memory" / "instruction_generator" / "connectivity_mp3d"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MP3D connectivity JSON")
    parser.add_argument("--scene-id", type=str, default="Pm6F8kyY3z2")
    parser.add_argument("--connectivity-dir", type=Path, default=DEFAULT_CONN_DIR)
    parser.add_argument("--out-json", type=Path, default=None, help="可选：写入摘要 JSON")
    args = parser.parse_args()

    conn_dir = args.connectivity_dir if args.connectivity_dir.is_absolute() else RAG4VLN_ROOT / args.connectivity_dir
    if not conn_dir.is_dir():
        raise SystemExit(f"目录不存在: {conn_dir}")

    candidates = sorted(conn_dir.glob(f"{args.scene_id}*_connectivity*.json"))
    if not candidates:
        raise SystemExit(f"未找到 {args.scene_id!r} 的 connectivity 文件于 {conn_dir}")

    path = candidates[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"期望 list，得到 {type(data)}")

    included = sum(1 for x in data if isinstance(x, dict) and x.get("included", True))
    sample_keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []

    summary = {
        "file": str(path),
        "scene_id_arg": args.scene_id,
        "num_nodes": len(data),
        "num_included": included,
        "sample_field_keys": sample_keys,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.out_json is not None:
        outp = args.out_json.expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[visualize_connectivity_mp3d] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
