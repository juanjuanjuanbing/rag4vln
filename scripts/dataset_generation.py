#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 VLN-CE raw_data 目录生成派生数据集：改写 instruction_text，并将 instruction_tokens 置为 null。
每条 episode 的 instruction 内会额外写入 original_instruction_text（生成前的原文），便于对照；评测仍只使用 instruction_text。

在 InternNav 仓库根目录执行示例：

  python rag4vln/scripts/dataset_generation.py \\
    --input-root data/vln_ce/raw_data \\
    --output-root data/vln_ce/raw_data_mask \\
    --method semantic_drop \\
    --drop-ratio 0.5 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# 侧车字段：保存改写前的指令（Habitat 一般忽略未知键）
ORIGINAL_INSTRUCTION_TEXT_KEY = "original_instruction_text"

# 必须用捕获组，split 才会把标点留在 tokens 里，才能拼回原句
_SEG_CAPTURE_RE = re.compile(r"([,，。\.！？!?；;]+)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def split_instruction_segments(text: str) -> List[str]:
    """
    按中英文标点切分为若干块；每块为「该段文字 + 紧跟其后的标点」（最后一块可能没有尾部标点）。
    拼接时用 "" 连接即可还原标点与段间空格。
    """
    s = text or ""
    if not s.strip():
        return []
    tokens = _SEG_CAPTURE_RE.split(s)
    segments: List[str] = []
    buf = ""
    for i, t in enumerate(tokens):
        if i % 2 == 0:
            buf = t
        else:
            combined = buf + t
            if combined.strip():
                segments.append(combined)
            buf = ""
    if buf and buf.strip():
        segments.append(buf)
    return segments


def transform_semantic_drop(text: str, drop_ratio: float, rng: random.Random) -> str:
    """
    按标点切成短块（每块自带其后标点）；最后一块永不删。
    - drop_ratio <= 0：不删，返回原文。
    - drop_ratio >= 1：只保留最后一块。
    - 否则：对其余每块以概率 drop_ratio 独立丢弃。
    合并时直接拼接，保留原有标点与空格。
    """
    if drop_ratio <= 0:
        return text
    parts = split_instruction_segments(text)
    if len(parts) <= 1:
        return parts[0] if parts else (text or "")

    last = parts[-1]
    prefix = parts[:-1]

    if drop_ratio >= 1.0:
        kept_prefix: List[str] = []
    else:
        kept_prefix = [p for p in prefix if rng.random() >= drop_ratio]

    merged = "".join(kept_prefix + [last]).strip()
    return merged if merged else last.strip()


def transform_goal_transform(text: str, **_kwargs: Any) -> str:
    """目标转换（预留接口，当前原样返回）。"""
    # TODO: 接入目标改写 / 替换等逻辑
    return text


def _load_json(path: Path) -> Dict[str, Any]:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _iter_dataset_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    yield from sorted(root.rglob("*.json"))
    for p in sorted(root.rglob("*.gz")):
        if p.name.endswith(".json.gz"):
            yield p


def _process_episodes(
    data: Dict[str, Any],
    transform: Callable[..., str],
    transform_kw: Dict[str, Any],
) -> tuple[int, int]:
    """返回 (处理的 episode 数, 跳过无 instruction 的数)。"""
    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        return 0, 0

    n_ok = 0
    n_skip = 0
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        instr = ep.get("instruction")
        if not isinstance(instr, dict):
            n_skip += 1
            continue
        text = instr.get("instruction_text")
        if not isinstance(text, str):
            n_skip += 1
            continue
        instr[ORIGINAL_INSTRUCTION_TEXT_KEY] = text
        instr["instruction_text"] = transform(text, **transform_kw)
        instr["instruction_tokens"] = None
        n_ok += 1
    return n_ok, n_skip


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="VLN-CE raw_data 派生数据集生成")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/vln_ce/raw_data"),
        help="源数据根目录（相对仓库根或绝对路径），将递归处理其中 .json / .json.gz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="输出根目录（相对仓库根或绝对路径），目录结构与 input-root 下相对路径一致",
    )
    parser.add_argument(
        "--method",
        choices=("semantic_drop", "goal_transform"),
        default="semantic_drop",
        help="semantic_drop：按标点分句随机删（最后一句保留）；goal_transform：预留，当前不改写",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=0.5,
        help="semantic_drop：对「非最后一句」每句以该概率删除；0=不删；1=只保留最后一句；范围建议 [0,1]",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（semantic_drop）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将处理的文件，不写盘",
    )
    args = parser.parse_args(argv)

    repo = _repo_root()
    in_root = args.input_root if args.input_root.is_absolute() else (repo / args.input_root).resolve()
    out_root = args.output_root if args.output_root.is_absolute() else (repo / args.output_root).resolve()

    if not in_root.is_dir():
        raise SystemExit(f"input-root 不是目录: {in_root}")

    dr = float(args.drop_ratio)
    if dr < 0 or dr > 1:
        raise SystemExit("--drop-ratio 应在 [0, 1] 内")

    rng = random.Random(int(args.seed))

    if args.method == "semantic_drop":

        def _tf(t: str, **_k: Any) -> str:
            return transform_semantic_drop(t, dr, rng)

        transform_kw: Dict[str, Any] = {}
    else:
        _tf = transform_goal_transform
        transform_kw = {}

    files = list(_iter_dataset_files(in_root))
    if not files:
        print(f"[dataset_generation] 未找到 json/json.gz: {in_root}", file=sys.stderr)
        return

    print(f"[dataset_generation] input={in_root}\n[dataset_generation] output={out_root}", flush=True)
    print(f"[dataset_generation] method={args.method} drop_ratio={dr} seed={args.seed}", flush=True)

    total_ep = 0
    total_skip = 0
    for src in files:
        try:
            rel = src.relative_to(in_root)
        except ValueError:
            continue
        dst = out_root / rel
        if args.dry_run:
            print(f"  would process: {rel}")
            continue

        data = _load_json(src)
        n_ok, n_skip = _process_episodes(data, _tf, transform_kw)
        total_ep += n_ok
        total_skip += n_skip
        _save_json(dst, data)
        print(f"  wrote {rel} episodes_ok={n_ok} skipped={n_skip}", flush=True)

    if not args.dry_run:
        print(f"[dataset_generation] done episodes_transformed={total_ep} skipped={total_skip}", flush=True)


if __name__ == "__main__":
    main()
