"""図5: UDP送信・UE同時実行による推論時間の増加（torch.compile あり、中央値 ± MAD）を描く。

UDP ありの各行を、同じ手法・数値精度・実行設定の UDP なしの行と対応付けて比べる。
出力: udp_overhead.png と、対応付けた表 udp_overhead.csv

    python measurements/paper_figures/plot_udp_overhead.py [--csv paper_results.csv] [--out_dir results_png]
"""

import argparse
from pathlib import Path

from plot_common import (SERIES_COLORS, add_common_args, is_quantized, load_plot_rows,
                         np, pd, plt, save_figure, series_name)

MATCH_COLUMNS = ["method", "backbone", "dtype", "quantization_mode", "runtime_variant"]

# y 軸の中間（低い棒と高い棒の間）を圧縮する割合（軸全体を 1 とした位置）
AXIS_POS_BREAK_LO = 0.55   # 0..break_lo が軸の55%
AXIS_POS_BREAK_HI = 0.63   # 圧縮区間が8%
AXIS_POS_TOP = 1.00        # break_hi..y_max が37%


def _as_key(values):
    return values.where(values.notna(), "__NA__").astype(str)


def _scalar_key(value):
    return "__NA__" if (value is None or (isinstance(value, float) and pd.isna(value))) else str(value)


def build_udp_table(rows_no_udp, rows_udp):
    """UDP ありの各行に、対応する UDP なしの行の推論時間を並べた表を返す。"""
    no_udp_keys = {column: _as_key(rows_no_udp[column]) for column in MATCH_COLUMNS}
    table = []
    for _, udp_row in rows_udp.iterrows():
        matches = rows_no_udp["compile"] == bool(udp_row["compile"])
        for column in MATCH_COLUMNS:
            matches &= no_udp_keys[column] == _scalar_key(udp_row[column])
        baseline_rows = rows_no_udp[matches]
        if len(baseline_rows) == 0:
            continue
        baseline = baseline_rows.iloc[0]
        quantized = is_quantized(udp_row)
        quantization_tag = " / INT8" if quantized else ""
        compile_tag = " / compile" if udp_row["compile"] else " / no compile"
        precision = "INT8" if quantized else str(udp_row["dtype"])
        if udp_row["method"] == "ROMP":
            short_label = f"{udp_row['method']} ({udp_row['backbone']})\n({precision})"
        else:
            short_label = f"{udp_row['method']}\n({precision})"

        overhead_ms = udp_row["robust_infer_ms_median"] - baseline["robust_infer_ms_median"]
        table.append({
            "method":            udp_row["method"],
            "config":            f"{series_name(udp_row)} / {udp_row['dtype']}{quantization_tag}{compile_tag}",
            "short_label":       short_label,
            "compile":           bool(udp_row["compile"]),
            "latency_ms_no_udp": baseline["robust_infer_ms_median"],
            "latency_ms_udp":    udp_row["robust_infer_ms_median"],
            "mad_no_udp":        baseline["robust_infer_ms_mad"],
            "mad_udp":           udp_row["robust_infer_ms_mad"],
            "overhead_ms":       overhead_ms,
            "overhead_pct":      100 * overhead_ms / baseline["robust_infer_ms_median"],
        })
    return pd.DataFrame(table)


def _set_broken_y_axis(ax, low_tops_ms, high_tops_ms):
    """低い棒と高い棒の間の区間を圧縮した y 軸にする（棒は途切れない）。"""
    break_lo_ms = max(low_tops_ms) * 1.20
    break_hi_ms = min(high_tops_ms) * 0.92
    y_max_ms = max(high_tops_ms) * 1.15

    def forward(y):
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        below = y <= break_lo_ms
        between = (y > break_lo_ms) & (y < break_hi_ms)
        above = y >= break_hi_ms
        out[below] = (y[below] / break_lo_ms) * AXIS_POS_BREAK_LO
        out[between] = AXIS_POS_BREAK_LO + (y[between] - break_lo_ms) / (break_hi_ms - break_lo_ms) \
            * (AXIS_POS_BREAK_HI - AXIS_POS_BREAK_LO)
        out[above] = AXIS_POS_BREAK_HI + (y[above] - break_hi_ms) / (y_max_ms - break_hi_ms) \
            * (AXIS_POS_TOP - AXIS_POS_BREAK_HI)
        return out

    def inverse(v):
        v = np.asarray(v, dtype=float)
        out = np.empty_like(v)
        below = v <= AXIS_POS_BREAK_LO
        between = (v > AXIS_POS_BREAK_LO) & (v < AXIS_POS_BREAK_HI)
        above = v >= AXIS_POS_BREAK_HI
        out[below] = (v[below] / AXIS_POS_BREAK_LO) * break_lo_ms
        out[between] = break_lo_ms + (v[between] - AXIS_POS_BREAK_LO) / (AXIS_POS_BREAK_HI - AXIS_POS_BREAK_LO) \
            * (break_hi_ms - break_lo_ms)
        out[above] = break_hi_ms + (v[above] - AXIS_POS_BREAK_HI) / (AXIS_POS_TOP - AXIS_POS_BREAK_HI) \
            * (y_max_ms - break_hi_ms)
        return out

    ax.set_yscale("function", functions=(forward, inverse))
    ax.set_ylim(0, y_max_ms)
    low_ticks = [t for t in [0, 20, 40, 60, 80, 100] if t <= break_lo_ms]
    high_ticks = [t for t in [150, 200, 250, 300, 350, 400] if break_hi_ms <= t <= y_max_ms]
    ax.set_yticks(low_ticks + high_ticks)

    # 圧縮区間に折れ目マーク
    d = 0.008
    v_mid = (AXIS_POS_BREAK_LO + AXIS_POS_BREAK_HI) / 2
    kw = dict(transform=ax.transAxes, color="black", clip_on=False, lw=1.2)
    ax.plot((-d, d), (v_mid - d, v_mid + d), **kw)
    ax.plot((-d, d), (v_mid - d + 0.015, v_mid + d + 0.015), **kw)


def plot_udp_overhead(table, rows_udp):
    table = table[table["compile"]].reset_index(drop=True)
    n_configs = len(table)
    if n_configs == 0:
        fig, _ax = plt.subplots(figsize=(6, 4))
        return fig

    x = np.arange(n_configs)
    bar_width = 0.38
    bar_colors = [SERIES_COLORS[series_name(rows_udp[rows_udp["method"] == row["method"]].iloc[0])]
                  for _, row in table.iterrows()]
    error_style = {"elinewidth": 1.2, "capsize": 4, "capthick": 1.2, "ecolor": "black"}

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * n_configs + 3), 5.6))
    ax.bar(x - bar_width / 2, table["latency_ms_no_udp"], width=bar_width,
           yerr=table["mad_no_udp"], color=bar_colors, edgecolor="none", error_kw=error_style)
    ax.bar(x + bar_width / 2, table["latency_ms_udp"], width=bar_width,
           yerr=table["mad_udp"], color=bar_colors, edgecolor="black",
           linewidth=1.0, hatch="//", error_kw=error_style)

    tops_ms = np.concatenate([table["latency_ms_no_udp"].values, table["latency_ms_udp"].values])
    low_tops_ms = tops_ms[tops_ms <= 100]
    high_tops_ms = tops_ms[tops_ms > 100]
    if len(high_tops_ms) > 0 and len(low_tops_ms) > 0:
        _set_broken_y_axis(ax, low_tops_ms, high_tops_ms)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)

    ax.set_xticks(x)
    ax.set_xticklabels(table["short_label"], fontsize=18)
    ax.set_ylabel("inference time\n(ms / frame, median)", fontsize=19)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="none", label="no streaming"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black",
                      linewidth=1.0, hatch="//", label="with streaming"),
    ], loc="upper left", fontsize=18)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    args = parser.parse_args()
    rows_no_udp, rows_udp = load_plot_rows(args.csv)
    table = build_udp_table(rows_no_udp, rows_udp)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    table_path = Path(args.out_dir) / "udp_overhead.csv"
    table.to_csv(table_path, index=False)
    print(" ", table_path)
    save_figure(plot_udp_overhead(table, rows_udp), args.out_dir, "udp_overhead.png")


if __name__ == "__main__":
    main()
