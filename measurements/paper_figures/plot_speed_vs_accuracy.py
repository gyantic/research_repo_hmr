"""図2: 推論時間と3DPW精度（PA-MPJPE・MPJPE・PVE・PA-PVE）の散布図を描く。

torch.compile ありの行だけを描き、エラーバーは 2×MAD。
出力: speed_vs_<metric>__compile.png と、論文値の★付きの speed_vs_<metric>__compile_with_papervalue.png
（PA-PVE は論文値が無いので★付きは作らない）。

    python measurements/paper_figures/plot_speed_vs_accuracy.py [--csv paper_results.csv] [--out_dir results_png]
"""

import argparse

from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

from plot_common import (SERIES_COLORS, SERIES_ORDER, add_common_args, is_quantized,
                         load_plot_rows, np, pd, plt, save_figure)

METRICS = ["PA-MPJPE", "MPJPE", "PVE", "PA-PVE"]
METRICS_WITHOUT_PAPER_VALUE = {"PA-PVE"}
# 指標ごとに論文値の★を出さない手法（関節集合が本評価と異なるため）
PAPER_VALUE_EXCLUDE = {
    "PA-MPJPE": {"FastSAM3DBody"},
}

# マーカー・エラーバーのサイズ（ここを変えれば全図に効く）
MARKER_SIZE = 160
STAR_SIZE = 420
ERR_LW = 1.8
ERR_SCALE = 2.0     # エラーバーは 2×MAD で表示
ERR_CAPSIZE = 5
LEG_METHOD_SIZE = 8.5
LEG_DTYPE_SIZE = 7.5
LEG_STYLE_SIZE = 9
LEG_STAR_SIZE = 13
X_TICKS_MS = [5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500]


def _axes_setup(ax, *, metric, title, rows=None, has_stars=False):
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("inference time  (ms / frame, median)   —   ← faster", fontsize=19)
    ax.set_ylabel(f"{metric}  (mm)   —   more accurate →", fontsize=19)
    ax.set_title(title, fontsize=21)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=18)

    if rows is None or len(rows) == 0:
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000])
        ax.set_xticklabels(["10", "20", "50", "100", "200", "500", "1000"])
        return

    # データに合わせて軸範囲を決める
    xerr_ms = rows["robust_infer_ms_mad"].fillna(0)
    x_lo_ms = (rows["robust_infer_ms_median"] - xerr_ms).min()
    x_hi_ms = (rows["robust_infer_ms_median"] + xerr_ms).max()
    lo_log = np.log10(max(x_lo_ms, 1e-3))
    hi_log = np.log10(x_hi_ms)
    pad = 0.08 * (hi_log - lo_log + 0.3)
    ax.set_xlim(10 ** (lo_log - pad), 10 ** (hi_log + pad))

    ys = rows[metric].dropna().values.tolist()
    if has_stars:
        paper_column = f"paper_{metric}"
        if paper_column in rows.columns:
            ys += rows[paper_column].dropna().values.tolist()
    if ys:
        y_lo, y_hi = min(ys), max(ys)
        y_pad = 0.08 * (y_hi - y_lo + 1)
        ax.set_ylim(y_hi + y_pad, y_lo - y_pad)   # 反転軸

    xlim_lo, xlim_hi = ax.get_xlim()
    ticks = [t for t in X_TICKS_MS if xlim_lo <= t <= xlim_hi]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])


def _draw_points(ax, rows, metric):
    for _, row in rows.iterrows():
        y = row[metric]
        if pd.isna(y):
            continue
        x = row["robust_infer_ms_median"]
        color = SERIES_COLORS[row["series"]]
        if row["style"] == "compile_udp":
            face, edge, edge_width = color, "black", 1.8
        elif row["style"] == "compile_no_udp":
            face, edge, edge_width = color, color, 1.4
        else:
            face, edge, edge_width = "none", color, 1.4
        xerr = (row["robust_infer_ms_mad"] * ERR_SCALE) if pd.notna(row["robust_infer_ms_mad"]) else 0
        if xerr > 0:
            ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor=color,
                        elinewidth=ERR_LW, capsize=ERR_CAPSIZE,
                        capthick=ERR_LW, alpha=0.85, zorder=2)
        ax.scatter(x, y, marker=row["marker"], s=MARKER_SIZE,
                   facecolors=face, edgecolors=edge, linewidths=edge_width, zorder=3)


def _draw_stars(ax, rows, metric):
    """論文値（fp32・量子化なしの行の paper_<metric>）を★で描く。"""
    paper_column = f"paper_{metric}"
    if paper_column not in rows.columns:
        return
    excluded = PAPER_VALUE_EXCLUDE.get(metric, set())
    for _, row in rows.iterrows():
        if row["method"] in excluded or row["dtype"] != "fp32" or is_quantized(row):
            continue
        paper_value = row[paper_column]
        if pd.notna(paper_value):
            ax.scatter(row["robust_infer_ms_median"], paper_value,
                       marker="*", s=STAR_SIZE,
                       facecolors=SERIES_COLORS[row["series"]],
                       edgecolors="white", linewidths=0.8, zorder=4)


def _legend(ax, *, show_paper_value):
    handles = []
    for series in SERIES_ORDER:
        handles.append(Line2D([], [], marker="o", color="none",
                              markerfacecolor=SERIES_COLORS[series],
                              markeredgecolor=SERIES_COLORS[series],
                              linestyle="none",
                              markersize=LEG_METHOD_SIZE, label=series))
    handles.append(Line2D([], [], linestyle="none", label=""))
    handles.append(Line2D([], [], marker="o", color="gray", linestyle="none",
                          markersize=LEG_DTYPE_SIZE, label="fp32"))
    handles.append(Line2D([], [], marker="D", color="gray", linestyle="none",
                          markersize=LEG_DTYPE_SIZE, label="bf16"))
    handles.append(Line2D([], [], marker="^", color="gray", linestyle="none",
                          markersize=LEG_DTYPE_SIZE, label="INT8"))
    if show_paper_value:
        handles.append(Line2D([], [], linestyle="none", label=""))
        handles.append(Line2D([], [], marker="*", color="none",
                              markerfacecolor="gray", markeredgecolor="white",
                              linestyle="none", markersize=LEG_STAR_SIZE,
                              label="paper value"))
    ax.legend(handles=handles, loc="lower right", fontsize=17,
              framealpha=0.95, labelspacing=0.5)


def plot_compiled(rows_no_udp, metric, show_paper_value):
    rows = rows_no_udp[rows_no_udp["compile"] == True]
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    _draw_points(ax, rows, metric)
    if show_paper_value:
        _draw_stars(ax, rows, metric)
    _axes_setup(ax, metric=metric, title=f"Speed vs {metric} on 3DPW  (torch.compile)",
                rows=rows, has_stars=show_paper_value)
    _legend(ax, show_paper_value=show_paper_value)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    args = parser.parse_args()
    rows_no_udp, _rows_udp = load_plot_rows(args.csv)
    for metric in METRICS:
        variants = [False] if metric in METRICS_WITHOUT_PAPER_VALUE else [True, False]
        file_metric = metric.replace("-", "")
        for show_paper_value in variants:
            suffix = "_with_papervalue" if show_paper_value else ""
            save_figure(plot_compiled(rows_no_udp, metric, show_paper_value), args.out_dir,
                        f"speed_vs_{file_metric}__compile{suffix}.png")


if __name__ == "__main__":
    main()
