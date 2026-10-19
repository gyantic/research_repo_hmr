"""図3: SAT-HMR INT8 量子化に対する torch.compile 適用範囲ごとの推論時間（中央値 ± MAD）を描く。

出力: sathmr_int8_compile_scope.png

    python measurements/paper_figures/plot_sathmr_int8_compile_scope.py [--csv paper_results.csv] [--out_dir results_png]
"""

import argparse

from plot_common import add_common_args, load_ok_rows, np, plt, save_figure

# runtime_variant -> 図のラベル（上から表示する順）
SCOPE_LABELS = {
    "torch_compile_quantized_linear": "encoder + decoder + heads",
    "torch_compile_encoder_decoder":  "encoder + decoder",
    "torch_compile_encoder_only":     "encoder",
    "eager_no_int8_attn":             "no compile",
}


def plot_compile_scope(csv_path):
    # 図2・図5用の絞り込み（INT8 + compile は代表1点のみ）を通す前の行を使う
    rows = load_ok_rows(csv_path)
    rows = rows[
        (rows["method"] == "SAT-HMR")
        & (rows["quantization_mode"].astype(str).str.startswith("int8"))
        & (rows["communication_protocol"].isna())
        & (rows["runtime_variant"].isin(list(SCOPE_LABELS)))
    ].copy()
    rows = rows.set_index("runtime_variant").loc[list(SCOPE_LABELS)].reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(rows))
    ax.barh(y, rows["robust_infer_ms_median"],
            xerr=rows["robust_infer_ms_mad"],
            color="#1f77b4", edgecolor="none",
            error_kw={"elinewidth": 1.4, "capsize": 4, "capthick": 1.4,
                      "ecolor": "black"})
    for i, row in rows.iterrows():
        ax.text(row["robust_infer_ms_median"] + row["robust_infer_ms_mad"] + 5, i,
                f"{row['robust_infer_ms_median']:.1f} ± {row['robust_infer_ms_mad']:.2f}",
                va="center", fontsize=18)
    ax.set_yticks(y)
    ax.set_yticklabels([SCOPE_LABELS[v] for v in rows["runtime_variant"]], fontsize=18)
    ax.invert_yaxis()
    ax.set_xlabel("inference time (ms/frame, median)", fontsize=19)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, rows["robust_infer_ms_median"].max() * 1.25)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    args = parser.parse_args()
    save_figure(plot_compile_scope(args.csv), args.out_dir, "sathmr_int8_compile_scope.png")


if __name__ == "__main__":
    main()
