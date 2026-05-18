#!/usr/bin/env python3
"""
从 bench 输出的 latency 文本（如 my_latency_bge_vit.txt）读取「饼图用数据 2」稳态段，
绘制单次指令耗时的圆环图（不含 KB 冷启动）。

四块（毫秒，来自文件中的 *_avg）：
  Encoding       ← query_embed_avg
  Scoring        ← scoring_avg
  Path Retrieving ← postprocess_avg（output_with_path + index 等）
  Augmenting     ← augment_avg

图尺寸默认 8cm×8cm；仅绘制圆环扇区，不绘制任何文字；配色与检索网格注意力热图一致（紫→蓝→绿）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# 与 rag4vln/scripts/plot_retriever_grid_heatmaps.py 中热图一致
_ATTENTION_HEX = [
    "#4A148C",
    "#6A1B9A",
    "#7E57C2",
    "#5C6BC0",
    "#3949AB",
    "#1E88E5",
    "#039BE5",
    "#00838F",
    "#00897B",
    "#43A047",
    "#66BB6A",
    "#7CB342",
]


def _segment_colors(n: int = 4) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return [_ATTENTION_HEX[len(_ATTENTION_HEX) // 2]]
    idx = [round(i * (len(_ATTENTION_HEX) - 1) / (n - 1)) for i in range(n)]
    return [_ATTENTION_HEX[int(k)] for k in idx]


def parse_steady_state_ms(latency_txt: Path) -> dict[str, float]:
    """
    解析「饼图用数据 2：稳态（不含 KB）」表格：label + ms。
    """
    raw = latency_txt.read_text(encoding="utf-8")
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "饼图用数据 2" in line and "稳态" in line:
            start = i
            break
    if start is None:
        raise ValueError(f"未找到「饼图用数据 2」稳态段: {latency_txt}")

    j = start + 1
    while j < len(lines) and (lines[j].strip().startswith("#") or not lines[j].strip()):
        j += 1
    if j >= len(lines):
        raise ValueError(f"稳态段后无表头: {latency_txt}")
    header = lines[j].strip().split()
    if len(header) < 2 or header[0].lower() != "label":
        raise ValueError(f"期望表头 label ms，得到: {lines[j]!r}")

    out: dict[str, float] = {}
    j += 1
    while j < len(lines):
        line = lines[j].strip()
        j += 1
        if not line or line.startswith("#"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        key, val_s = parts[0], parts[1]
        try:
            out[key] = float(val_s)
        except ValueError:
            continue
    return out


def _steady_key_order() -> list[str]:
    """稳态四项在圆环中的顺序（与 Encoding → … → Augmenting 一致）。"""
    return [
        "query_embed_avg",
        "scoring_avg",
        "postprocess_avg",
        "augment_avg",
    ]


def _write_figure(fig: mpl.figure.Figure, out_stem: Path, pad_inches: float = 0.02) -> None:
    for ext, kw in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = out_stem.with_suffix(ext)
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=pad_inches,
            facecolor="white",
            format=ext[1:],
            **kw,
        )


def plot_donut(
    ms_by_key: dict[str, float],
    out_stem: Path,
    *,
    fig_cm: float,
    ring_width: float = 0.42,
) -> None:
    order = _steady_key_order()
    missing = [k for k in order if k not in ms_by_key]
    if missing:
        raise KeyError(f"稳态表格缺少键: {missing}；已有: {sorted(ms_by_key)}")

    values = [float(ms_by_key[k]) for k in order]
    colors = _segment_colors(len(values))

    mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})
    inch = fig_cm / 2.54
    fig = plt.figure(figsize=(inch, inch))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=ring_width, edgecolor="white", linewidth=0.8),
    )
    ax.set_aspect("equal")
    ax.axis("off")

    _write_figure(fig, out_stem, pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency 稳态圆环图（来自 my_latency_*.txt）")
    rag4vln_root = Path(__file__).resolve().parents[1]
    default_txt = rag4vln_root / "results" / "my_latency_bge_vit.txt"
    parser.add_argument("--latency-txt", type=Path, default=default_txt, help="bench 输出的 txt")
    parser.add_argument(
        "--out-stem",
        type=Path,
        default=None,
        help="输出路径无前缀（默认与 txt 同目录下的 latency_steady_donut）",
    )
    parser.add_argument("--fig-cm", type=float, default=8.0, help="图宽与高（厘米），1:1")
    parser.add_argument("--ring-width", type=float, default=0.42, help="圆环厚度（0–1，相对半径）")
    args = parser.parse_args()

    latency_txt: Path = args.latency_txt
    out_stem = args.out_stem or (latency_txt.parent / "latency_steady_donut")

    ms = parse_steady_state_ms(latency_txt)
    plot_donut(ms, out_stem, fig_cm=args.fig_cm, ring_width=args.ring_width)
    print(f"已写入: {out_stem}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
