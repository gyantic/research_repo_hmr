#!/usr/bin/env python3
"""Range plot for inference latency across precision/backend configs."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- data ---
pytorch = {
    "fp32":            {"min": 30.67, "median": 34.99, "mean": 35.00, "p95": 40.13, "max": 40.96},
    "bf16":            {"min": 26.44, "median": 28.74, "mean": 29.20, "p95": 32.00, "max": 40.80},
    "torchao\nweight-only INT8": {"min": 58.96, "median": 65.30, "mean": 67.02, "p95": 79.45, "max": 87.51},
    "torchao\ndynamic INT8":     {"min": 702.79,"median": 763.13,"mean": 788.07,"p95": 918.84,"max": 966.95},
}
trt = {
    "TRT FP16": {"mean": 15.96, "p95": 21.54},
    "TRT INT8\n(QDQ)": {"mean": 23.10, "p95": 44.16},
}

labels_pt  = list(pytorch.keys())
labels_trt = list(trt.keys())
all_labels = labels_pt + labels_trt
n = len(all_labels)

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title("SAT-HMR Inference Latency – Range Plot (batch=1, no UDP)", fontsize=13)
ax.set_ylabel("Latency (ms)")
ax.set_xlabel("Config")

x = np.arange(n)
width = 0.5
colors_pt  = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
colors_trt = ["#8172B2", "#937860"]

# PyTorch configs
for i, (label, d) in enumerate(pytorch.items()):
    c = colors_pt[i]
    # min-max range
    ax.bar(x[i], d["max"] - d["min"], bottom=d["min"],
           width=width, color=c, alpha=0.25, linewidth=0)
    # median line
    ax.hlines(d["median"], x[i]-width/2, x[i]+width/2, colors=c, linewidths=2.5, label="_nolegend_")
    # mean dot
    ax.plot(x[i], d["mean"], "o", color=c, markersize=6, zorder=5)
    # p95 tick
    ax.hlines(d["p95"], x[i]-width/3, x[i]+width/3, colors=c, linewidths=1.5, linestyles="--")

# TRT configs (mean + p95 only)
for j, (label, d) in enumerate(trt.items()):
    i = len(pytorch) + j
    c = colors_trt[j]
    ax.bar(x[i], d["p95"] - d["mean"], bottom=d["mean"],
           width=width, color=c, alpha=0.25, linewidth=0)
    ax.plot(x[i], d["mean"], "o", color=c, markersize=6, zorder=5)
    ax.hlines(d["p95"], x[i]-width/3, x[i]+width/3, colors=c, linewidths=1.5, linestyles="--")
    ax.annotate("(mean~p95\nonly)", xy=(x[i], d["p95"]+1), ha="center", fontsize=7, color=c)

ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_xlim(-0.6, n - 0.4)
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3, which="both")

# legend
legend_items = [
    mpatches.Patch(color="gray", alpha=0.3, label="min–max range"),
    plt.Line2D([0], [0], color="gray", linewidth=2.5, label="median"),
    plt.Line2D([0], [0], color="gray", marker="o", markersize=6, linewidth=0, label="mean"),
    plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="p95"),
]
ax.legend(handles=legend_items, fontsize=8, loc="upper left")

out = "plots/inference_range_plot.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
