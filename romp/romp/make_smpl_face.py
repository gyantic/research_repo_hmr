# make_smpl_face.py  ── 一回だけ実行: SMPL の face 配列を smpl_faces.npy に書き出す
import torch, numpy as np, pathlib

pth_path = pathlib.Path(__file__).resolve().parents[2]/"body_models"/"smpl"/"SMPL_NEUTRAL.pth"
if not pth_path.exists():
    raise FileNotFoundError(f"{pth_path} が見つかりません。READMEの手順で SMPL_NEUTRAL.pth を作成してください。")

smpl = torch.load(pth_path, map_location="cpu")   # 1～2 秒
faces = smpl["f"].cpu().numpy().astype(np.int32)  # (13776, 3)

out = pathlib.Path(__file__).with_name("smpl_faces.npy")
np.save(out, faces)
print(f"{out} を書き出しました  shape={faces.shape}")
