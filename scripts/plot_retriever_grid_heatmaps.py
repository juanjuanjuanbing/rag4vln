#!/usr/bin/env python3
"""
从 retriever 网格评估的 summary.csv 生成热图（仅 Matplotlib）。

默认读取 grid_a0_b10_k2_20260419_201134/summary.csv；绘制刻度与数值标签，不绘制 alpha/beta 等轴名称。
指标：scene_hit@1、view_start_hit@1、view_end.match_score（自 metrics.json）。

主图 6cm×6cm、1:1；色标 [0,1] 不画在主图；单独导出 colorbar（默认竖条高约 5cm）。
除 PNG 外同时导出 PDF / SVG 矢量图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def _reference_cmap() -> LinearSegmentedColormap:
    """紫 → 蓝 → 绿：低值偏紫，高值偏绿（热图与 colorbar 共用）。"""
    return LinearSegmentedColormap.from_list(
        "purple_blue_green",
        [
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
        ],
        N=512,
    )


def _apply_style(font_pt: float) -> LinearSegmentedColormap:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": font_pt,
            "axes.labelsize": font_pt,
            "axes.titlesize": font_pt,
            "xtick.labelsize": font_pt,
            "ytick.labelsize": font_pt,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    return _reference_cmap()


def _load_frame(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df = df[df["status"] == "ok"].copy()
    match_scores: list[float] = []
    for p in df["metrics_path"]:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        match_scores.append(float(m["overall"]["view_end"]["match_score"]))
    df["goal_match_score"] = match_scores
    return df


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    alphas = sorted(df["alpha"].unique())
    betas = sorted(df["beta"].unique())
    piv = df.pivot(index="beta", columns="alpha", values=value_col)
    return piv.reindex(index=betas[::-1], columns=alphas)


def _pivot_extent(piv: pd.DataFrame) -> tuple[float, float, float, float]:
    """imshow extent: (left, right, bottom, top)，origin=upper 时首行对应 top。"""
    alphas = piv.columns.astype(float).to_numpy()
    betas = piv.index.astype(float).to_numpy()
    da = float((alphas[1] - alphas[0]) / 2) if len(alphas) > 1 else 0.05
    db = float(abs(betas[0] - betas[1]) / 2) if len(betas) > 1 else 0.05
    left, right = float(alphas[0] - da), float(alphas[-1] + da)
    bottom, top = float(betas[-1] - db), float(betas[0] + db)
    return (left, right, bottom, top)


def _write_figure(fig: mpl.figure.Figure, out_stem: Path, pad_inches: float = 0.02) -> None:
    """同一画布写入 PNG / PDF / SVG（stem 无后缀，如 .../heatmap_scene_at1）。"""
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


def _fmt_tick(v: float) -> str:
    """避免 0.20000000000000004 这类显示。"""
    return f"{float(v):g}"


def save_heatmap(
    piv: pd.DataFrame,
    out_stem: Path,
    cmap: mpl.colors.Colormap,
    fig_cm: float,
    font_pt: float,
) -> None:
    inch = fig_cm / 2.54
    fig, ax = plt.subplots(figsize=(inch, inch), constrained_layout=True)
    arr = piv.to_numpy(dtype=float)
    extent = _pivot_extent(piv)
    ax.imshow(
        arr,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        extent=extent,
        aspect="equal",
        interpolation="bicubic",
    )
    alphas = piv.columns.astype(float).to_numpy()
    betas_display = sorted(float(x) for x in piv.index)
    ax.set_xticks(alphas)
    ax.set_xticklabels([_fmt_tick(x) for x in alphas])
    ax.set_yticks(betas_display)
    ax.set_yticklabels([_fmt_tick(x) for x in betas_display])
    ax.tick_params(
        axis="both",
        which="major",
        left=True,
        bottom=True,
        labelleft=True,
        labelbottom=True,
        labelsize=font_pt,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.6)
    _write_figure(fig, out_stem, pad_inches=0.02)
    plt.close(fig)


def save_colorbar_only(
    cmap: mpl.colors.Colormap,
    out_stem: Path,
    font_pt: float,
    height_cm: float,
) -> None:
    """竖直 colorbar：height_cm 为色条方向长度；宽度按旧版 (0.15in / 1.6in) 比例缩放。"""
    h_in = height_cm / 2.54
    w_in = h_in * (0.15 / 1.6)
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=font_pt)
    _write_figure(fig, out_stem, pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retriever grid heatmaps (matplotlib only)")
    rag4vln_root = Path(__file__).resolve().parents[1]
    default_grid = (
        rag4vln_root
        / "results"
        / "retriever_eval_grid"
        / "grid_a0_b10_k2_20260419_201134"
        / "summary.csv"
    )
    parser.add_argument("--summary-csv", type=Path, default=default_grid, help="summary.csv 路径")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认同 summary 所在 grid 文件夹）",
    )
    parser.add_argument("--fig-cm", type=float, default=6.0, help="图宽与高（厘米），1:1")
    parser.add_argument("--font-pt", type=float, default=10.0, help="刻度与全局字号（pt）")
    parser.add_argument(
        "--colorbar-height-cm",
        type=float,
        default=5.0,
        help="单独 colorbar 图在竖直方向的高度（厘米）",
    )
    args = parser.parse_args()

    summary_csv: Path = args.summary_csv
    out_dir: Path = args.out_dir or summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = _apply_style(args.font_pt)
    df = _load_frame(summary_csv)

    specs = [
        ("scene_hit@1", "heatmap_scene_at1"),
        ("view_start_hit@1", "heatmap_view_at1"),
        ("goal_match_score", "heatmap_goal_match_score"),
    ]

    for col, stem in specs:
        piv = _pivot(df, col)
        save_heatmap(piv, out_dir / stem, cmap, args.fig_cm, args.font_pt)

    save_colorbar_only(
        cmap,
        out_dir / "colorbar_0_1_vertical",
        args.font_pt,
        args.colorbar_height_cm,
    )
    print(f"已写入: {out_dir}")


if __name__ == "__main__":
    main()
