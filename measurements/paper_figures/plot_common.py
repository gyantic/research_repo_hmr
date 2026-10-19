"""論文の図（図2・図3・図5）を描くスクリプトの共通部分。

入力は paper_results.csv（cp932）。status が ok の行だけを使う。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "paper_results.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results_png"

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 19,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "axes.titlesize": 20,
})

NUMERIC_COLUMNS = ["robust_infer_ms_median", "robust_infer_ms_mad",
                   "MPJPE", "PA-MPJPE", "PVE", "PA-PVE",
                   "paper_MPJPE", "paper_PA-MPJPE", "paper_PVE"]

SERIES_COLORS = {
    "ROMP (HRNet32)":  "#1f5fa8",
    "ROMP (ResNet50)": "#8fc1e8",
    "SAT-HMR":         "#1f9e6e",
    "FastSAM3DBody":   "#e0552a",
}
SERIES_ORDER = ["ROMP (HRNet32)", "ROMP (ResNet50)", "SAT-HMR", "FastSAM3DBody"]


def add_common_args(parser):
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="paper_results.csv のパス")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="出力先（無ければ作成）")


def load_ok_rows(csv_path):
    """status が ok の行を、数値列を数値に変換して返す（絞り込みなし）。"""
    df = pd.read_csv(csv_path, encoding="cp932")
    df = df[df["status"] == "ok"].copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def is_quantized(row):
    quantization = str(row["quantization_mode"])
    return quantization not in ("none", "nan")


def series_name(row):
    if row["method"] == "ROMP":
        return f"ROMP ({row['backbone']})"
    return row["method"]


def _keep_row(row):
    # no-compile + UDP は除外。SAT-HMR INT8 + compile は代表1点（全部位をコンパイル）のみ残す
    if (not row["compile"]) and row["communication_protocol"] == "UDP":
        return False
    if row["method"] == "SAT-HMR" and is_quantized(row) and row["compile"]:
        return row["runtime_variant"] == "torch_compile_quantized_linear"
    return True


def _dtype_marker(row):
    if is_quantized(row):
        return "^"
    return "D" if row["dtype"] == "bf16" else "o"


def _style_category(row):
    udp = row["communication_protocol"] == "UDP"
    if row["compile"] and udp:
        return "compile_udp"
    if row["compile"]:
        return "compile_no_udp"
    return "nocompile_no_udp"


def load_plot_rows(csv_path):
    """図2・図5用の行を返す: (UDPなしの行, UDPありの行)。"""
    df = load_ok_rows(csv_path)
    df = df[df.apply(_keep_row, axis=1)].copy()
    df["series"] = df.apply(series_name, axis=1)
    df["marker"] = df.apply(_dtype_marker, axis=1)
    df["style"] = df.apply(_style_category, axis=1)
    is_udp = df["communication_protocol"] == "UDP"
    return df[~is_udp].copy(), df[is_udp].copy()


def save_figure(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(" ", path)


__all__ = ["np", "pd", "plt"]
