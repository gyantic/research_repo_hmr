"""論文で使用した、量子化などで作成した重みを Hugging Face から取得し、各モデルが読み込む場所へ配置する。

Usage:
    python download_weights.py            # 全モデル
    python download_weights.py sathmr     # モデルを指定（sathmr / romp / fastsam）
    python download_weights.py --force    # 既存ファイルも取得し直す

非公開リポジトリの場合は環境変数 HF_TOKEN にアクセストークンを設定する。
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# 配布先の Hugging Face リポジトリ（<ユーザー名>/<リポジトリ名>）
REPO_ID = "kimurakmkm123/research_repo_hmr"
REVISION = "main"

REPO_ROOT = Path(__file__).resolve().parent

# リポジトリルートからの相対パス。Hugging Face 側にも同じパスで置く。
WEIGHTS = {
    "sathmr": [
        "sat-hmr/weights/sat_hmr/sat_644_bf16.pth",
        "sat-hmr/weights/sat_hmr/sat_644_int8w.pth",
        "sat-hmr/weights/sat_hmr/sat_644_3dpw_int8w.pth",
    ],
    "romp": [
        "romp/trained_models/ROMP_HRNet32_V1_bf16.pth",
        "romp/trained_models/ROMP_ResNet50_V1_bf16.pth",
        "romp/trained_models/ROMP_HRNet32_V1_hrnet32_backbone_int8wo.pth",
        "romp/trained_models/ROMP_ResNet50_V1_int8_raw_weights.pth",
    ],
    "fastsam": [
        "fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_int8.engine",
        "fast-sam-3d-body/checkpoints/moge_trt/moge_dinov2_encoder_int8.engine",
        "fast-sam-3d-body/checkpoints/yolo/yolo11m-pose.engine",
        "fast-sam-3d-body/checkpoints/bf16/model_bf16.ckpt",
        "fast-sam-3d-body/checkpoints/bf16/mhr_model_bf16.pt",
        "fast-sam-3d-body/checkpoints/bf16/model_bf16.pt",
    ],
}


def download(repo_id, path, force):
    dest = REPO_ROOT / path
    if dest.exists() and not force:
        print(f"[SKIP] {path}")
        return
    url = f"https://huggingface.co/{repo_id}/resolve/{REVISION}/{path}"
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    print(f"[GET ] {path}")
    with urllib.request.urlopen(req) as res, open(tmp, "wb") as f:
        while chunk := res.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=[], help="取得するモデル（sathmr / romp / fastsam。省略時は全部）")
    parser.add_argument("--repo", default=REPO_ID, help="Hugging Face のリポジトリ名")
    parser.add_argument("--force", action="store_true", help="既存ファイルも取得し直す")
    args = parser.parse_args()
    unknown = set(args.models) - set(WEIGHTS)
    if unknown:
        parser.error(f"未知のモデル: {', '.join(sorted(unknown))}（sathmr / romp / fastsam から選ぶ）")
    if not args.repo:
        sys.exit("[ERROR] 配布先が未設定です。download_weights.py の REPO_ID か --repo で指定してください。")
    for model in args.models or WEIGHTS:
        for path in WEIGHTS[model]:
            download(args.repo, path, args.force)


if __name__ == "__main__":
    main()
