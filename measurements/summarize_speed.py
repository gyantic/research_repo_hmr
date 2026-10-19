#!/usr/bin/env python3
"""速度計測の生データ（フレーム別CSV）から、推論時間・総処理時間の中央値と中央絶対偏差（MAD）を出す。

論文の Table 1・図2・図3・図5 の値（中央値 ± MAD）はこの計算による。MAD は |x - 中央値| の中央値（尺度補正なし）。
既定では各CSVの先頭10フレームを除く（初回コンパイルを含むウォームアップの扱いに合わせる）。

読めるCSVの形式:
  - romp_video_demo.py / video_demo.py / benchmark_fastsam3dbody.py : 列 inference_ms, total_ms
  - measurements/romp/benchmark_romp_udp_ue.py : 「# 全フレームデータ」以降の inference_time_ms, total_time_ms
  - run_publisher_ue.py の --metrics-csv : 列 infer_ms, total_ms

    python measurements/summarize_speed.py results/measurements/raw_*.csv [--skip_rows 10] [--output summary.csv]
"""

import argparse
import csv
import io
import statistics
from pathlib import Path

# 形式ごとの (推論時間の列, 総処理時間の列)
TIME_COLUMNS = [
    ("inference_ms", "total_ms"),
    ("inference_time_ms", "total_time_ms"),
    ("infer_ms", "total_ms"),
]
ROMP_UDP_DATA_MARKER = "# 全フレームデータ"


def read_frame_rows(path):
    """フレームごとの行を辞書で返す。ROMP UDP 計測の統計サマリー部分は読み飛ばす。"""
    text = Path(path).read_text(encoding="utf-8-sig")
    if ROMP_UDP_DATA_MARKER in text:
        text = text.split(ROMP_UDP_DATA_MARKER, 1)[1]
    lines = [line for line in text.splitlines() if line.strip()]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def column_values_ms(rows, column):
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def median_and_mad(values_ms):
    if not values_ms:
        return float("nan"), float("nan")
    median = statistics.median(values_ms)
    return median, statistics.median(abs(v - median) for v in values_ms)


def summarize(path, skip_rows):
    rows = read_frame_rows(path)
    if not rows:
        raise SystemExit(f"No frame rows in {path}")
    inference_column, total_column = next(
        (columns for columns in TIME_COLUMNS if columns[0] in rows[0]), (None, None))
    if inference_column is None:
        raise SystemExit(f"No known timing column in {path}: {list(rows[0])}")
    rows = rows[skip_rows:]
    inference_median, inference_mad = median_and_mad(column_values_ms(rows, inference_column))
    total_median, total_mad = median_and_mad(column_values_ms(rows, total_column))
    return {
        "raw_csv": str(path),
        "frames": len(rows),
        "skipped_rows": skip_rows,
        "inference_ms_median": inference_median,
        "inference_ms_mad": inference_mad,
        "total_ms_median": total_median,
        "total_ms_mad": total_mad,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_csv", nargs="+")
    parser.add_argument("--skip_rows", type=int, default=10, help="先頭から除くフレーム数")
    parser.add_argument("--output", help="結果を書くCSV（省略時は表示のみ）")
    args = parser.parse_args()

    summaries = [summarize(path, args.skip_rows) for path in args.raw_csv]
    for s in summaries:
        print(f"{Path(s['raw_csv']).name}: frames={s['frames']}  "
              f"inference {s['inference_ms_median']:.2f} ± {s['inference_ms_mad']:.2f} ms  "
              f"total {s['total_ms_median']:.2f} ± {s['total_ms_mad']:.2f} ms")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print("saved:", args.output)


if __name__ == "__main__":
    main()
