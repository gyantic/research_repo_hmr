"""download_weights.py に載っている重みを、hf_weights/ のモデルカード・ライセンス文と一緒に Hugging Face へアップロードする。

Usage:
    hf auth login                         # 初回のみ
    python upload_weights.py              # 全モデル
    python upload_weights.py sathmr       # モデルを指定（sathmr / romp / fastsam）

huggingface_hub が必要（sathmr-copy・sam_3d_body 環境には導入済み）。
"""

import argparse
import sys

from huggingface_hub import HfApi

from download_weights import REPO_ID, REPO_ROOT, WEIGHTS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=[], help="アップロードするモデル（sathmr / romp / fastsam。省略時は全部）")
    parser.add_argument("--repo", default=REPO_ID, help="Hugging Face のリポジトリ名")
    parser.add_argument("--private", action="store_true", help="リポジトリを非公開で作成する")
    args = parser.parse_args()
    unknown = set(args.models) - set(WEIGHTS)
    if unknown:
        parser.error(f"未知のモデル: {', '.join(sorted(unknown))}（sathmr / romp / fastsam から選ぶ）")
    if not args.repo:
        sys.exit("[ERROR] 配布先が未設定です。download_weights.py の REPO_ID か --repo で指定してください。")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    # モデルカード（README.md）とライセンス文（licenses/）をリポジトリ直下へ置く
    print("[PUT ] hf_weights/ -> README.md, licenses/")
    api.upload_folder(folder_path=str(REPO_ROOT / "hf_weights"), repo_id=args.repo, repo_type="model")
    for model in args.models or WEIGHTS:
        for path in WEIGHTS[model]:
            local = REPO_ROOT / path
            if not local.exists():
                print(f"[MISS] {path}")
                continue
            print(f"[PUT ] {path}")
            api.upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=args.repo, repo_type="model")


if __name__ == "__main__":
    main()
