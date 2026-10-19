import os.path as _osp
_repo_root = _osp.dirname(_osp.dirname(_osp.dirname(_osp.abspath(__file__))))
# データセットはリポジトリ直下の datasets/ に置く（3DPW は datasets/3dpw/。ROMP・Fast-SAM-3D-Body と共用）
dataset_root = _osp.join(_repo_root, 'datasets')
# SMPL はリポジトリ直下の body_models/smpl/ に1か所だけ置く（ROMP・Fast-SAM-3D-Body と共用）
smpl_model_path = _osp.join(_repo_root, 'body_models')
smpl_mean_path = _osp.join(smpl_model_path, 'smpl', 'smpl_mean_params.npz')

dinov2_vitb14_path = './weights/dinov2/dinov2_vitb14_pretrain.pth'
dinov2_vitl14_path = './weights/dinov2/dinov2_vitl14_pretrain.pth'
