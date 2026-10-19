"""ROMP の 3DPW 精度評価を実行し、結果を JSON に保存する（図2の ROMP）。

評価は ROMP の romp.test と同じ処理で、表示される結果（MPJPE・PA-MPJPE・PVE・PA-PVE など）を
キーそのままで保存する。romp/ ディレクトリから、ROMP の README と同じ PYTHONPATH で実行する:

    cd romp
    PYTHONPATH="$PWD:$PWD/romp" python ../measurements/romp/evaluate_3dpw.py \
        --configs_yml configs/h36m14_hrnet_fp32.yml --output_json ../results/measurements/accuracy_romp_hrnet_fp32.json
"""
import argparse
import json
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_json", required=True)
    args, romp_argv = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + romp_argv
    from romp.test import ConfigContext, Demo
    with ConfigContext():
        eval_results = Demo().test_eval()

    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({name: float(value) for name, value in eval_results.items()}, indent=2))
    print("saved:", output_json)


if __name__ == "__main__":
    main()
