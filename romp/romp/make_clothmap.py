# make_clothmap.py ← 一度だけ実行: 衣装メッシュ各頂点の SMPL 4近傍スキンウェイトを作る
import os
import numpy as np, trimesh, torch
from sklearn.neighbors import NearestNeighbors

# ① SMPL テンプレ体 (6890 頂点)
SMPL_PTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'body_models', 'smpl', 'SMPL_NEUTRAL.pth')
smpl = torch.load(open(SMPL_PTH,'rb'), encoding='latin1')
V_tmpl = smpl['v_template']                # (6890,3)

# ② 衣装メッシュ (OBJ)
cloth = trimesh.load('ROMP/romp/Medieval01_Unfold3d.obj', process=False)
V_c   = cloth.vertices                     # (M,3)

# ③ 最近接 4 点で重み (ほぼ 0.2 ms 用)
nbrs = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(V_tmpl)
d, idx = nbrs.kneighbors(V_c, return_distance=True)
w = 1.0 / (d + 1e-8)                       # inverse-distance
w /= w.sum(1,keepdims=True)                # 正規化

np.save('cloth_idx4.npy', idx.astype(np.int32))   # (M,4)
np.save('cloth_w4.npy',   w.astype(np.float32))   # (M,4)
print('saved skin weights', w.shape)
