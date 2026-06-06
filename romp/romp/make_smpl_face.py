# make_smpl_faces.py  ── 一回だけ実行
import torch, numpy as np, pathlib, os

pth_path = pathlib.Path.home()/".romp/SMPL_NEUTRAL.pth"
if not pth_path.exists():
    raise FileNotFoundError(f"{pth_path} が見つかりません。ROMP を一度起動してモデルをダウンロードしてください。")

smpl = torch.load(pth_path, map_location="cpu")   # 1～2 秒
faces = smpl["f"].cpu().numpy().astype(np.int32)  # (13776, 3)

out = pathlib.Path(__file__).with_name("smpl_faces.npy")
np.save(out, faces)
print(f"{out} を書き出しました  shape={faces.shape}")
