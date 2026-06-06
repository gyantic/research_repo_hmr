"""
SAT-HMR Video Demo
動画ファイルを入力として、3Dメッシュ推定を行い、結果を動画として出力します。
"""

"""
args：
-i 入力動画ファイル名(これは必須)
-o 出力動画ファイル名(ないなら入力名_sathmr.mp4)
--conf_thresh：検出信頼度閾値 (デフォルト 0.3)
--no_save 保存しないならTrue
"""

import argparse
import atexit
import csv
import logging
import os
import socket
import sys
import random
import time
from types import MethodType

# --- xformers無効化 / INT8 attention有効化の環境変数セット ---
# models import前に必ず立てる必要があるため argparse より前で sys.argv を peek する。
_ARGV = sys.argv[1:]
def _ptq_mode_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--ptq_mode" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--ptq_mode="):
            return a.split("=", 1)[1]
    if "--torchao_int8" in argv:
        return "int8_dynamic"
    return "none"
_PTQ_MODE_PEEK = _ptq_mode_from_argv(_ARGV)
_PTQ_ACTIVE = _PTQ_MODE_PEEK != "none"
_NO_INT8_ATTN = "--no_int8_attn" in _ARGV
if "--xformers_off" in _ARGV:
    os.environ.setdefault("XFORMERS_DISABLED", "1")
if _NO_INT8_ATTN:
    os.environ.pop("SAT_HMR_INT8_ATTENTION", None)
elif "--int8_attn" in _ARGV:
    os.environ.setdefault("SAT_HMR_INT8_ATTENTION", "1")


def split_torch_logs_out_per_run():
    log_path = os.environ.get("TORCH_LOGS_OUT")
    if not log_path:
        return None
    root, ext = os.path.splitext(log_path)
    if not ext:
        ext = ".log"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    split_path = f"{root}.{timestamp}.{os.getpid()}{ext}"
    parent = os.path.dirname(split_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    os.environ["TORCH_LOGS_OUT"] = split_path
    return split_path


_SPLIT_TORCH_LOGS_OUT = split_torch_logs_out_per_run()


def _argv_value(argv, *names):
    for i, arg in enumerate(argv):
        for name in names:
            if arg == name and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return None


def _run_log_dir_from_argv(argv):
    log_dir = _argv_value(argv, "--log_dir")
    if log_dir:
        return log_dir
    output_dir = _argv_value(argv, "--output_dir")
    if output_dir:
        return output_dir
    for path in (
        _argv_value(argv, "--metrics_csv"),
        _argv_value(argv, "--pred_npz"),
        _argv_value(argv, "--output", "-o"),
    ):
        if not path:
            continue
        path_dir = os.path.dirname(path)
        if path_dir and path_dir != os.path.sep:
            return path_dir
    return os.path.join("plots", "run_logs")


def dequantize_int8_weight_state_dict_if_needed(state_dict):
    int8_keys = [k for k in state_dict if k.endswith(".weight_int8")]
    if not int8_keys:
        return state_dict, {"loaded_int8_weight_file": False, "dequantized_tensors": 0}

    restored = {}
    for key, tensor in state_dict.items():
        if key.endswith(".weight_int8"):
            base = key[: -len("_int8")]
            scale_key = base + "_scale"
            if scale_key not in state_dict:
                raise KeyError(f"Missing INT8 weight scale for {key}: expected {scale_key}")
            scale = state_dict[scale_key].float().unsqueeze(1)
            restored[base] = tensor.float() * scale
        elif key.endswith(".weight_scale"):
            continue
        else:
            restored[key] = tensor

    return restored, {
        "loaded_int8_weight_file": True,
        "dequantized_tensors": len(int8_keys),
    }


class TeeStream:
    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, data):
        self.primary.write(data)
        if not self.log_file.closed:
            self.log_file.write(data)

    def flush(self):
        self.primary.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.primary, name)


def setup_run_log(argv):
    log_dir = _run_log_dir_from_argv(argv)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"video_demo.{timestamp}.{os.getpid()}.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    atexit.register(log_file.flush)
    print(f"[INFO] run log: {log_path}")
    return log_path


_RUN_LOG_PATH = setup_run_log(_ARGV)

import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def patch_inductor_same_meta():
    import torch._inductor.fx_passes.post_grad as pg

    orig_same_meta = pg.same_meta

    def safe_same_meta(node1, node2):
        val1 = node1.meta.get("val")
        val2 = node2.meta.get("val")
        if val1 is None or val2 is None:
            return False
        if not hasattr(val1, "size") or not hasattr(val2, "size"):
            return False
        return orig_same_meta(node1, node2)

    pg.same_meta = safe_same_meta


patch_inductor_same_meta()
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms

from models import build_sat_model
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img, vis_meshes_img, get_colors_rgb

SEED = 42
DTYPE_DEBUG_ONCE = {
    'model': False,
    'img': False,
    'smpl': False,
}


def setup_reproducibility(seed: int = 42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


setup_reproducibility(SEED)


TORCH_COMPILE_LOGGER_NAMES = (
    "torch._dynamo",
    "torch._inductor",
    "torch.fx.experimental.symbolic_shapes",
)


class _SuppressTorchCompileConsoleFilter(logging.Filter):
    def filter(self, record):
        name = record.name
        return not any(
            name == logger_name or name.startswith(f"{logger_name}.")
            for logger_name in TORCH_COMPILE_LOGGER_NAMES
        )


def suppress_torch_compile_console_logs():
    compile_filter = _SuppressTorchCompileConsoleFilter()
    installed = 0
    for logger_name in ("torch",) + TORCH_COMPILE_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            if getattr(handler, "stream", None) is not sys.stderr:
                continue
            if getattr(handler, "_sat_compile_filter_installed", False):
                continue
            handler.addFilter(compile_filter)
            handler._sat_compile_filter_installed = True
            installed += 1
    return installed


def build_compile_options(compile_mode, compile_cudagraphs):
    options = {}
    if compile_cudagraphs == "on":
        options["triton.cudagraphs"] = True
        options["triton.cudagraph_support_input_mutation"] = True
    elif compile_cudagraphs == "off":
        options["triton.cudagraphs"] = False
    elif compile_mode == "max-autotune-no-cudagraphs":
        options["triton.cudagraphs"] = False
    return options


def resolve_compile_call(compile_mode, compile_options):
    if not compile_options:
        return compile_mode, None

    merged_options = {}
    if compile_mode and compile_mode != "none":
        try:
            merged_options.update(torch._inductor.list_mode_options(compile_mode))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to expand compile mode '{compile_mode}' into options "
                "for this torch version."
            ) from exc
    merged_options.update(compile_options)
    return None, merged_options


def wrap_quantized_linears_for_compile(model):
    wrapped = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = getattr(module, "weight", None)
        if type(weight) in (torch.Tensor, torch.nn.Parameter):
            continue
        if getattr(module, "_sat_compile_eager_wrapped", False):
            continue
        module.forward = MethodType(torch._dynamo.disable(module.forward.__func__), module)
        module._sat_compile_eager_wrapped = True
        wrapped.append(name)
    return wrapped


def get_args():
    parser = argparse.ArgumentParser('SAT-HMR Video Demo')
    parser.add_argument('--input', '-i', required=True, help='入力動画ファイルパス') #"/videos/sample_video.mp4"
    parser.add_argument('--output', '-o', default=None, help='出力動画ファイルパス')
    parser.add_argument('--output_dir', default=None, help='出力ファイルを置くディレクトリ')
    parser.add_argument('--log_dir', default=None, help='実行ログを置くディレクトリ')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='検出信頼度閾値')
    parser.add_argument('--weights', default='./weights/sat_hmr/sat_644.pth', help='重みファイルパス')
    parser.add_argument('--input_size', type=int, default=1288, help='入力画像サイズ')
    parser.add_argument('--batch_size', type=int, default=1, help='推論バッチサイズ（デフォルト: 1）')
    parser.add_argument('--device', default='cuda', help='デバイス')
    parser.add_argument("--dtype_cfg", default="FP32", help="変数の型(fp32, bfloat16, int8)")
    parser.add_argument('--torchao_int8', action='store_true', help='[後方互換] --ptq_mode int8_dynamic の alias')
    parser.add_argument('--ptq_mode',
                        choices=['none', 'int8_dynamic', 'int8_weight_only', 'int4_weight_only'],
                        default='none',
                        help='torchao PTQ スキーム')
    parser.add_argument('--int8_skip_modules', nargs='*', default=[],
                        help='int8_dynamic 量子化から外す module 名パターン')
    parser.add_argument('--int4_group_size', type=int, default=128, help='INT4 weight-only の group_size')
    parser.add_argument('--int4_skip_modules', nargs='*', default=[], help='INT4量子化スキップする module 名パターン')
    parser.add_argument('--xformers_off', action='store_true', help='XFORMERS_DISABLED=1 を setdefault')
    parser.add_argument('--int8_attn', action='store_true', help='SAT_HMR_INT8_ATTENTION=1 を setdefault')
    parser.add_argument('--no_int8_attn', action='store_true', help='SAT_HMR_INT8_ATTENTIONを無効化')
    parser.add_argument('--compile', action='store_true', help='torch.compile(model) を有効化')
    parser.add_argument('--compile_mode', default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                        help='torch.compile の mode')
    parser.add_argument('--compile_cudagraphs', default='auto',
                        choices=['auto', 'on', 'off'],
                        help='Inductor CUDAGraphs の制御')
    parser.add_argument('--compile_dynamic', action='store_true',
                        help='torch.compile で dynamic shape を許可 (default: static)')
    parser.add_argument('--compile_quantized_linear', action='store_true',
                        help='量子化済み Linear も compile 対象に含める (default: eager に逃がす)')
    parser.add_argument('--compile_disable_encoder', action='store_true',
                        help='encoder 部分を torch.compile 対象から外す')
    parser.add_argument('--compile_decoder_only', action='store_true',
                        help='model 全体ではなく decoder のみを torch.compile する')
    parser.add_argument('--compile_encoder_only', action='store_true',
                        help='model 全体ではなく encoder (forward_encoder) のみを torch.compile する')
    parser.add_argument('--compile_heads', action='store_true',
                        help='pose_head/shape_head/cam_head/conf_head/bbox_embed をそれぞれ個別に torch.compile する')
    parser.add_argument('--warmup_frames', type=int, default=0,
                        help='平均処理時間から最初の N フレームを除外する (compile warmup amortize 用)')
    parser.add_argument('--prewarm_frames', type=int, default=0,
                        help='実処理前に N フレーム空回しする (UDP送信/metrics記録なし)')
    parser.add_argument('--prewarm_after_hello', action='store_true',
                        help='UDP HELLO を待ってから空回しする (デフォルト: HELLO 待ち前)')
    parser.add_argument('--no_prewarm_rewind', action='store_true',
                        help='動画入力でも空回し後に先頭へ戻さない')
    parser.add_argument('--static_int8_fused_mlp', action='store_true',
                        help='[experimental] encoder DINO MLP / decoder FFN を static W8A8 fused Triton kernel に置換する')
    parser.add_argument('--static_int8_calibration_frames', type=int, default=8,
                        help='static int8 系 (--static_int8_fused_mlp / --static_scale_int8_decoder_ffn) の calibration に使う frame 数')
    parser.add_argument('--static_scale_int8_decoder_ffn', action='store_true',
                        help='decoder の linear1/linear2 を compile-native static-scale INT8 Linear に置換する (Triton 不使用、Inductor fusion 維持)')
    parser.add_argument('--profile_forward', action='store_true',
                        help='model forward 内の encoder/decoder/head/SMPL 時間をログ出力')
    parser.add_argument('--profile_forward_frames', type=int, default=10,
                        help='--profile_forward で計測する forward 回数')
    parser.add_argument('--profile_encoder_blocks', action='store_true',
                        help='--profile_forward 時に encoder block ごとの時間もログ出力')
    parser.add_argument('--show_fps', action='store_true', help='FPS/処理時間を画面に重畳表示')
    parser.add_argument('--display', action='store_true', help='画面表示を有効化（デフォルト: 無効）')
    parser.add_argument('--no_save', action='store_true', help='保存しない')
    parser.add_argument('--metrics_csv', default=None, help='フレーム毎の測定CSV出力パス')
    parser.add_argument('--pred_npz', default=None, help='フレーム毎の予測NPZ出力パス（pseudo-GT比較用）')
    parser.add_argument('--no_render', action='store_true', help='メッシュ描画スキップ')
    parser.add_argument('--udp_only', action='store_true', help='UE送信専用モード（描画/表示を無効化）')
    parser.add_argument('--udp', action='store_true', help='UEへUDP送信を有効化')
    parser.add_argument('--ue_port', type=int, default=62004, help='UEの受信ポート')
    parser.add_argument('--local_port', type=int, default=5555, help='Python側待受ポート')
    parser.add_argument('--udp_track_max_dist_cm', type=float, default=120.0, help='追跡継続する最大距離(cm)')
    parser.add_argument('--udp_reinit_conf', type=float, default=0.55, help='追跡再初期化に必要な信頼度')
    parser.add_argument('--udp_hold_frames', type=int, default=10, help='見失い時に追跡を維持するフレーム数')
    parser.add_argument('--udp_trans_jump_cm', type=float, default=30.0, help='平行移動の外れ値閾値(cm)。<=0で無効')
    parser.add_argument('--udp_outlier_mode', choices=['skip', 'clamp'], default='skip', help='外れ値処理: skip=送信しない, clamp=移動量を閾値に制限')
    parser.add_argument('--udp_outlier_max_skip', type=int, default=12, help='skipモードで連続スキップ可能な最大フレーム数')
    parser.add_argument('--udp_jump_conf_thresh', type=float, default=0.35, help='ジャンプ判定の最小conf')
    parser.add_argument('--udp_jump_motion_cm', type=float, default=20.0, help='ジャンプ判定の移動量閾値(cm)')
    parser.add_argument('--udp_jump_pose_motion', type=float, default=0.02, help='ジャンプ判定のpose_motion閾値')
    parser.add_argument('--udp_jump_debounce', type=int, default=2, help='ジャンプ判定の連続成立フレーム数')
    parser.add_argument('--udp_jump_hold_frames', type=int, default=4, help='ジャンプ確定後に外れ値抑制をバイパスするフレーム数')
    parser.add_argument('--udp_jump_cooldown', type=int, default=6, help='ジャンプ終了後に再判定を抑制するフレーム数')
    return parser.parse_args()


def resolve_output_path(path, output_dir, label, root_filename_dir=None):
    if not path:
        return path
    path_dir = os.path.dirname(path)
    is_root_filename = os.path.isabs(path) and path_dir == os.path.sep
    if output_dir and (not path_dir or is_root_filename):
        fixed_path = os.path.join(output_dir, os.path.basename(path))
        if fixed_path != path:
            print(f"[INFO] {label}: {fixed_path}")
        return fixed_path
    if is_root_filename:
        target_dir = root_filename_dir or os.getcwd()
        fixed_path = os.path.join(target_dir, os.path.basename(path))
        print(f"[WARN] {label}: {path} はルート直下を指すため、出力先ディレクトリに変更します: {fixed_path}")
        return fixed_path
    return path


def load_config(input_size):
    demo_cfg_path = os.path.join('configs', 'run', 'demo.yaml')
    with open(demo_cfg_path) as f:
        demo_config = yaml.safe_load(f)
    
    model_name = demo_config.get('model', 'default')
    model_cfg_path = os.path.join('configs', 'models', f'{model_name}.yaml')
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)
    
    class Config:
        pass
    
    args = Config()
    for k, v in model_config.items():
        setattr(args, k, v)
    for k, v in demo_config.items():
        setattr(args, k, v)
    
    args.mode = 'infer'
    args.pretrain = False
    args.pretrain_path = None
    args.input_size = input_size
    
    return args


def process_img_like_base(img, input_size):
    """BASEクラスのprocess_imgと同じ処理"""
    ori_h, ori_w = img.shape[:2]
    
    # 長い辺をinput_sizeに合わせる
    if ori_w >= ori_h:
        resize_rate = input_size / ori_w
        new_w = input_size
        new_h = int(resize_rate * ori_h)
    else:
        resize_rate = input_size / ori_h
        new_h = input_size
        new_w = int(resize_rate * ori_w)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    img_size = (new_h, new_w)
    
    return img_resized, img_size, resize_rate


def preprocess_frame(frame, input_size, use_sat=True):
    """フレームを前処理（BASEクラスの__getitem__と同じ処理）"""
    # BGR画像をそのまま使用（process_imgと同じ）
    img_resized, img_size, resize_rate = process_img_like_base(frame, input_size)
    
    # パディング（__getitem__と同じ）
    patch_size = 56 if use_sat else 14
    pad_h = (img_size[0] + patch_size - 1) // patch_size * patch_size
    pad_w = (img_size[1] + patch_size - 1) // patch_size * patch_size
    
    pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    pad_img[:img_size[0], :img_size[1]] = img_resized
    
    # BGR -> RGB, PIL変換, 正規化
    pil_img = Image.fromarray(pad_img[:, :, ::-1].copy())
    
    array2tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    norm_img = array2tensor(pil_img)
    return norm_img, img_size, frame.shape[:2]


def smpl_pose_to_ue_quaternions(pose):
    rotations = Rotation.from_rotvec(pose.reshape(-1, 3))
    quaternions = rotations.as_quat().astype(np.float32)
    quaternions[:, 0] *= -1.0
    quaternions[:, 2] *= -1.0
    flip_root = np.array([1., 0., 0., 0.], dtype=np.float32)
    quaternions[0] = (
        Rotation.from_quat(flip_root) * Rotation.from_quat(quaternions[0])
    ).as_quat().astype(np.float32)
    return quaternions


def transl_to_ue(transl):
    trans = transl.copy().astype(np.float32)
    trans[1] *= -1.0
    trans[2] = 0.0
    return trans * 100.0


def select_udp_target(cands, tracker_state, max_dist_cm, reinit_conf):
    if len(cands['confs']) == 0:
        return None, tracker_state, None

    confs = cands['confs']
    poses = cands['poses']
    transls = cands['transls']
    query_ids = cands['query_ids']

    best_idx = int(np.argmax(confs))
    reason = "max_conf"

    if tracker_state['active'] and tracker_state['transl'] is not None:
        dists_cm = np.linalg.norm(transls - tracker_state['transl'][None, :], axis=1) * 100.0
        near_idx = int(np.argmin(dists_cm))
        near_dist = float(dists_cm[near_idx])
        near_conf = float(confs[near_idx])
        best_conf = float(confs[best_idx])

        if near_dist <= max_dist_cm:
            best_idx = near_idx
            reason = f"nearest:{near_dist:.1f}cm"
        elif best_conf >= reinit_conf:
            reason = f"reinit_conf:{best_conf:.3f}"
        else:
            return None, tracker_state, f"skip_jump nearest={near_dist:.1f}cm conf={near_conf:.3f}"

    selected = {
        'pose': poses[best_idx],
        'transl': transls[best_idx],
        'conf': float(confs[best_idx]),
        'query_id': int(query_ids[best_idx]),
        'num_persons': len(confs),
    }

    tracker_state['active'] = True
    tracker_state['transl'] = transls[best_idx].copy()
    tracker_state['query_id'] = int(query_ids[best_idx])
    tracker_state['missed'] = 0
    return selected, tracker_state, reason


def _postprocess_and_render(
    model,
    outputs,
    idx,
    frame,
    img_tensor,
    img_size,
    orig_size,
    input_size,
    conf_thresh,
    device,
    no_render=False,
    dtype="FP32",
):
    """1サンプル分の後処理と描画"""
    t0 = time.time()
    confs = outputs['pred_confs'][idx].squeeze(-1)
    select_idx = torch.where(confs > conf_thresh)[0]
    num_persons = int(select_idx.numel())

    udp_candidates = {
        'poses': np.empty((0, 72), dtype=np.float32),
        'transls': np.empty((0, 3), dtype=np.float32),
        'confs': np.empty((0,), dtype=np.float32),
        'query_ids': np.empty((0,), dtype=np.int32),
    }
    pred_record = {
        'joints2d': np.zeros((0, 24, 2), dtype=np.float32),
        'joints3d': np.zeros((0, 24, 3), dtype=np.float32),
        'verts': np.zeros((0, 0, 3), dtype=np.float32),
        'transl': np.zeros((0, 3), dtype=np.float32),
        'bboxes': np.zeros((0, 4), dtype=np.float32),
        'focal_length': np.zeros((0,), dtype=np.float32),
    }

    pred_verts = np.zeros((0, 0, 3), dtype=np.float32)
    if 'pred_intrinsics' in outputs:
        pred_intrinsics = outputs['pred_intrinsics'][idx].reshape(3, 3).detach().cpu()
    else:
        focal = input_size / (2.0 * np.tan(np.pi / 6.0))
        pred_intrinsics = torch.tensor(
            [
                [focal, 0.0, img_size[1] / 2.0],
                [0.0, focal, img_size[0] / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    #pred_intrinsics_np = pred_intrinsics.numpy().astype(np.float32)
    pred_intrinsics_np = pred_intrinsics.float().cpu().numpy().astype(np.float32)

    if num_persons > 0:
        poses_t = outputs['pred_poses'][idx][select_idx].detach().cpu().float().numpy().astype(np.float32)
        betas_t = outputs['pred_betas'][idx][select_idx].detach().cpu().float().numpy().astype(np.float32)
        transl_t = outputs['pred_transl'][idx][select_idx].detach().cpu().float().numpy().astype(np.float32)
        boxes_t = outputs['pred_boxes'][idx][select_idx].detach().cpu().float().numpy().astype(np.float32)
        conf_t = confs[select_idx].detach().cpu().float().numpy().astype(np.float32)
        qids_t = select_idx.detach().cpu().numpy().astype(np.int32)
        udp_candidates = {
            'poses': poses_t,
            'transls': transl_t,
            'confs': conf_t,
            'query_ids': qids_t,
        }

        with torch.no_grad():
            use_bf16 = str(dtype).lower() in {"bfloat16", "bf16", "bfp16"}
            tensor_dtype = torch.bfloat16 if use_bf16 else torch.float32
            poses_dev = torch.from_numpy(poses_t).to(device=device, dtype=tensor_dtype, non_blocking=True)
            betas_dev = torch.from_numpy(betas_t).to(device=device, dtype=tensor_dtype, non_blocking=True)
            transl_dev = torch.from_numpy(transl_t).to(device=device, dtype=tensor_dtype, non_blocking=True)
            if not DTYPE_DEBUG_ONCE['smpl']:
                print(f"[DTYPE] poses_dev dtype: {poses_dev.dtype}, betas_dev dtype: {betas_dev.dtype}")
                DTYPE_DEBUG_ONCE['smpl'] = True
            verts_smpl, joints_smpl = model.human_model(poses=poses_dev, betas=betas_dev)
            verts_cam = verts_smpl + transl_dev.unsqueeze(1)
            joints_cam = joints_smpl + transl_dev.unsqueeze(1)
            pred_verts = verts_cam.detach().cpu().float().numpy().astype(np.float32)
            j3d_t = joints_cam[:, :24, :].detach().cpu().float().numpy().astype(np.float32)

        j2d_t = np.zeros((num_persons, 24, 2), dtype=np.float32)
        z = np.clip(j3d_t[..., 2:3], 1e-6, None)
        j2d_t[..., 0] = j3d_t[..., 0] * pred_intrinsics_np[0, 0] / z[..., 0] + pred_intrinsics_np[0, 2]
        j2d_t[..., 1] = j3d_t[..., 1] * pred_intrinsics_np[1, 1] / z[..., 0] + pred_intrinsics_np[1, 2]

        pred_record = {
            'joints2d': j2d_t,
            'joints3d': j3d_t,
            'verts': pred_verts,
            'transl': transl_t,
            'bboxes': boxes_t,
            'focal_length': np.full((num_persons,), float(pred_intrinsics_np[0, 0]), dtype=np.float32),
        }
    postprocess_time = time.time() - t0

    if no_render:
        return frame, num_persons, {'postprocess': postprocess_time}, udp_candidates, pred_record

    # 描画
    t0 = time.time()
    
    # 正規化を戻した画像を取得
    ori_img = tensor_to_BGR(unNormalize(img_tensor).cpu())
    
    # メッシュを描画
    if num_persons > 0:
        colors = get_colors_rgb(num_persons)
        result_img = vis_meshes_img(
            img=ori_img.copy(),
            verts=pred_verts,
            smpl_faces=model.human_model.faces,
            cam_intrinsics=pred_intrinsics,
            colors=colors,
            padding=True
        )
    else:
        result_img = pad_img(ori_img, max(ori_img.shape[0], ori_img.shape[1]))
    
    # 有効領域を切り出し
    result_img = result_img[:img_size[0], :img_size[1]]
    
    # 元のサイズにリサイズ
    result_img = cv2.resize(result_img, (orig_size[1], orig_size[0]))

    render_time = time.time() - t0
    return result_img, num_persons, {'postprocess': postprocess_time, 'render': render_time}, udp_candidates, pred_record


def process_frame(model, frame, input_size, conf_thresh, device, no_render=False, dtype="FP32"):
    """1フレームを処理"""
    times = {}
    use_bf16 = str(dtype).lower() in {"bfloat16", "bf16", "bfp16"}

    # 前処理
    t0 = time.time()
    img_tensor, img_size, orig_size = preprocess_frame(frame, input_size, use_sat=True)
    img_tensor = img_tensor.to(device)
    if use_bf16:
        img_tensor = img_tensor.to(torch.bfloat16)
    if not DTYPE_DEBUG_ONCE['img']:
        print(f"[DTYPE] img_tensor dtype: {img_tensor.dtype}")
        DTYPE_DEBUG_ONCE['img'] = True

    targets = [{
        'img_size': torch.tensor(img_size).to(device),
        'img_path': 'video_frame'
    }]
    times['preprocess'] = time.time() - t0

    # 推論
    t0 = time.time()
    with torch.no_grad():
        outputs = model([img_tensor], targets)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times['inference'] = time.time() - t0

    result_frame, num_persons, extra_times, udp_candidates, pred_record = _postprocess_and_render(
        model=model,
        outputs=outputs,
        idx=0,
        frame=frame,
        img_tensor=img_tensor,
        img_size=img_size,
        orig_size=orig_size,
        input_size=input_size,
        conf_thresh=conf_thresh,
        device=device,
        no_render=no_render,
        dtype=dtype,
    )
    times.update(extra_times)
    return result_frame, num_persons, times, udp_candidates, pred_record


def process_frames_batch(model, frames, input_size, conf_thresh, device, no_render=False, dtype="FP32"):
    """複数フレームを1バッチで推論"""
    use_bf16 = str(dtype).lower() in {"bfloat16", "bf16", "bfp16"}
    batch_items = []
    for frame in frames:
        t0 = time.time()
        img_tensor, img_size, orig_size = preprocess_frame(frame, input_size, use_sat=True)
        img_tensor = img_tensor.to(device)
        if use_bf16:
            img_tensor = img_tensor.to(torch.bfloat16)
        if not DTYPE_DEBUG_ONCE['img']:
            print(f"[DTYPE] img_tensor dtype: {img_tensor.dtype}")
            DTYPE_DEBUG_ONCE['img'] = True
        target = {
            'img_size': torch.tensor(img_size).to(device),
            'img_path': 'video_frame'
        }
        preprocess_time = time.time() - t0
        batch_items.append({
            'frame': frame,
            'img_tensor': img_tensor,
            'img_size': img_size,
            'orig_size': orig_size,
            'target': target,
            'preprocess_time': preprocess_time,
        })

    if not batch_items:
        return []

    # バッチ推論
    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            [item['img_tensor'] for item in batch_items],
            [item['target'] for item in batch_items]
        )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time_each = (time.time() - t0) / len(batch_items)

    results = []
    for idx, item in enumerate(batch_items):
        result_frame, num_persons, extra_times, udp_candidates, pred_record = _postprocess_and_render(
            model=model,
            outputs=outputs,
            idx=idx,
            frame=item['frame'],
            img_tensor=item['img_tensor'],
            img_size=item['img_size'],
            orig_size=item['orig_size'],
            input_size=input_size,
            conf_thresh=conf_thresh,
            device=device,
            no_render=no_render,
            dtype=dtype,
        )
        times = {
            'preprocess': item['preprocess_time'],
            'inference': inference_time_each,
        }
        times.update(extra_times)
        results.append((result_frame, num_persons, times, udp_candidates, pred_record))

    return results


def run_prewarm_frames(cap, model, args, device, dtype, total_frames):
    """実処理前に forward/postprocess を空回しする。UDP送信・metrics記録は行わない。"""
    prewarm_n = max(0, int(getattr(args, 'prewarm_frames', 0) or 0))
    if prewarm_n <= 0:
        return 0

    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    can_rewind = total_frames > 0 and start_pos >= 0
    do_rewind = can_rewind and not getattr(args, 'no_prewarm_rewind', False)

    print(
        f"[PREWARM] start: frames={prewarm_n}, "
        f"rewind={'ON' if do_rewind else 'OFF'}, udp=OFF, metrics=OFF"
    )

    done = 0
    for idx in range(prewarm_n):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] prewarm stopped early: {done}/{prewarm_n} frame(s)")
            break

        _result_frame, num_persons, times, _udp_candidates, _pred_record = process_frame(
            model,
            frame,
            args.input_size,
            args.conf_thresh,
            device,
            no_render=args.no_render,
            dtype=dtype,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        done += 1

        if done == 1 or done == prewarm_n or done % 5 == 0:
            print(
                f"[PREWARM] {done}/{prewarm_n} "
                f"persons={num_persons} inference={times['inference'] * 1000.0:.1f}ms"
            )

    if do_rewind:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        print(f"[PREWARM] rewind to frame {int(start_pos)}")
    elif getattr(args, 'no_prewarm_rewind', False):
        print("[PREWARM] rewind skipped by --no_prewarm_rewind")
    else:
        print("[PREWARM] rewind skipped (stream/camera input)")

    print(f"[PREWARM] complete: {done}/{prewarm_n} frame(s)")
    return done


def main():
    args = get_args()

    if args.input_size % 56 != 0:
        print(f"エラー: input_size ({args.input_size}) は56の倍数である必要があります")
        sys.exit(1)
    if args.batch_size < 1:
        print("エラー: batch_size は1以上で指定してください")
        sys.exit(1)
    if not args.display:
        args.no_render = True
    if args.udp_only:
        args.no_render = True
        args.display = False
        args.show_fps = False
    dtype = args.dtype_cfg #bfloat16
    use_bf16 = str(dtype).lower() in {"bfloat16", "bf16", "bfp16"}

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"エラー: 動画を開けません: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    print(f"入力動画: {args.input}")
    print(f"解像度: {width}x{height}, FPS: {fps:.2f}, フレーム数: {total_frames}")

    if args.output is None:
        input_base = os.path.basename(args.input) if args.output_dir else args.input
        base = os.path.splitext(input_base)[0]
        args.output = f"{base}_sathmr.mp4"
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    args.output = resolve_output_path(args.output, args.output_dir, 'output')
    output_side_dir = os.path.dirname(args.output) or args.output_dir or os.getcwd()
    args.metrics_csv = resolve_output_path(args.metrics_csv, args.output_dir, 'metrics_csv', output_side_dir)
    args.pred_npz = resolve_output_path(args.pred_npz, args.output_dir, 'pred_npz', output_side_dir)

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"出力動画: {args.output}")

    print("モデルを読み込み中...")
    model_args = load_config(args.input_size)
    print(
        f"[INFO] SAT cfg: conf_thresh={model_args.sat_cfg.get('conf_thresh')} "
        f"scale_thresh={model_args.sat_cfg.get('scale_thresh')} "
        f"get_map_layer={model_args.sat_cfg.get('get_map_layer')} "
        f"num_lvls={model_args.sat_cfg.get('num_lvls')}"
    )
    model, _ = build_sat_model(model_args, set_criterion=False)
    

    print(f"重みを読み込み中: {args.weights}")
    state_dict = torch.load(args.weights, map_location='cpu', weights_only=False)
    state_dict, int8_weight_info = dequantize_int8_weight_state_dict_if_needed(state_dict)
    if int8_weight_info["loaded_int8_weight_file"]:
        print(
            "[INFO] Loaded INT8 weight file: "
            f"dequantized_tensors={int8_weight_info['dequantized_tensors']}"
        )
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if use_bf16:
        model = model.to(torch.bfloat16)
    # 後方互換: 旧 --torchao_int8 フラグを --ptq_mode int8_dynamic にマップ
    if args.torchao_int8 and args.ptq_mode == 'none':
        args.ptq_mode = 'int8_dynamic'

    # env_check（量子化適用前に1回）
    try:
        from models.encoders.dinov2.layers.attention import XFORMERS_AVAILABLE as _XA
    except Exception:
        _XA = None
    print(f"[ENV] XFORMERS_DISABLED={os.environ.get('XFORMERS_DISABLED')} "
          f"SAT_HMR_INT8_ATTENTION={os.environ.get('SAT_HMR_INT8_ATTENTION')} "
          f"XFORMERS_AVAILABLE(attention)={_XA}")

    if args.ptq_mode != 'none':
        try:
            from engines.engine import apply_torchao_int8, apply_torchao_int4
            if args.ptq_mode == 'int8_dynamic':
                qcfg_name = apply_torchao_int8(
                    model,
                    weight_only=False,
                    skip_modules=tuple(args.int8_skip_modules),
                )
            elif args.ptq_mode == 'int8_weight_only':
                qcfg_name = apply_torchao_int8(model, weight_only=True)
            elif args.ptq_mode == 'int4_weight_only':
                if not use_bf16:
                    model = model.to(torch.bfloat16)
                qcfg_name = apply_torchao_int4(
                    model,
                    skip_modules=tuple(args.int4_skip_modules),
                    group_size=args.int4_group_size,
                )
            print(f"[INFO] PTQ applied: {args.ptq_mode} ({qcfg_name})")
        except Exception as e:
            print(f"[ERROR] PTQ apply failed (mode={args.ptq_mode}): {e}")
            sys.exit(1)

    if hasattr(model, "prepare_static_eval_buffers"):
        try:
            with torch.no_grad():
                model.prepare_static_eval_buffers()
            print("[INFO] Static eval buffers prepared before compile")
        except Exception as e:
            print(f"[WARN] prepare_static_eval_buffers failed before compile: {e}")

    # Static W8A8 fused MLP/FFN replacement (after PTQ, before compile)
    if args.static_int8_fused_mlp:
        try:
            from models.static_int8_fused import (
                StaticInt8FusedMLP,
                CalibrationHooks,
                enumerate_hot_targets,
                apply_static_int8_fused,
            )
            if not StaticInt8FusedMLP.is_supported():
                print("[WARN] Triton not available, skipping --static_int8_fused_mlp")
            else:
                hot_targets = enumerate_hot_targets(model)
                print(f"[INFO] static_int8_fused_mlp: {len(hot_targets)} hot targets identified")
                if hot_targets:
                    # --- Calibration ---
                    n_calib = max(1, args.static_int8_calibration_frames)
                    print(f"[INFO] static_int8_fused_mlp: calibrating with {n_calib} frame(s)")
                    _calib_cap = cv2.VideoCapture(args.input)
                    calib_frames = []
                    for _ in range(n_calib):
                        ok, frm = _calib_cap.read()
                        if not ok:
                            break
                        calib_frames.append(frm)
                    _calib_cap.release()
                    if not calib_frames:
                        print("[WARN] no calibration frames available, skipping")
                    else:
                        # model.eval() is called later; force eval mode for calibration to skip training-only paths (DN, prepare_for_cdn)
                        prev_training = model.training
                        model.eval()
                        with torch.no_grad(), CalibrationHooks(hot_targets) as hooks:
                            for frm in calib_frames:
                                img_tensor, img_size, orig_size = preprocess_frame(frm, args.input_size, use_sat=True)
                                img_tensor = img_tensor.to(device)
                                if use_bf16:
                                    img_tensor = img_tensor.to(torch.bfloat16)
                                tgts = [{
                                    'img_size': torch.tensor(img_size).to(device),
                                    'img_path': 'calibration',
                                }]
                                _ = model([img_tensor], tgts)
                            scales_dict = hooks.scales()
                        if prev_training:
                            model.train()
                        # --- Replacement ---
                        n_replaced = apply_static_int8_fused(model, scales_dict)
                        scale_preview = ", ".join(
                            f"{k.split('/')[-1]}=({s[0]:.4g},{s[1]:.4g})"
                            for k, s in list(scales_dict.items())[:3]
                        )
                        print(
                            f"[INFO] static_int8_fused_mlp: replaced {n_replaced}/{len(hot_targets)} modules "
                            f"(sample scales: {scale_preview})"
                        )
        except Exception as e:
            import traceback
            print(f"[ERROR] static_int8_fused_mlp setup failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    # L3: compile-native static-scale INT8 decoder FFN replacement
    if args.static_scale_int8_decoder_ffn:
        try:
            from models.static_int8_fused import (
                enumerate_decoder_ffn_linears,
                LinearCalibrationHooks,
                apply_static_scale_int8_decoder_ffn,
            )
            ffn_targets = enumerate_decoder_ffn_linears(model)
            print(f"[INFO] static_scale_int8_decoder_ffn: {len(ffn_targets)} linear targets")
            if ffn_targets:
                n_calib = max(1, args.static_int8_calibration_frames)
                print(f"[INFO] static_scale_int8_decoder_ffn: calibrating with {n_calib} frame(s)")
                _calib_cap = cv2.VideoCapture(args.input)
                calib_frames = []
                for _ in range(n_calib):
                    ok, frm = _calib_cap.read()
                    if not ok:
                        break
                    calib_frames.append(frm)
                _calib_cap.release()
                if not calib_frames:
                    print("[WARN] no calibration frames available, skipping")
                else:
                    prev_training = model.training
                    model.eval()
                    with torch.no_grad(), LinearCalibrationHooks(ffn_targets) as hooks:
                        for frm in calib_frames:
                            img_tensor, img_size, orig_size = preprocess_frame(frm, args.input_size, use_sat=True)
                            img_tensor = img_tensor.to(device)
                            if use_bf16:
                                img_tensor = img_tensor.to(torch.bfloat16)
                            tgts = [{
                                'img_size': torch.tensor(img_size).to(device),
                                'img_path': 'calibration',
                            }]
                            _ = model([img_tensor], tgts)
                        ffn_scales = hooks.scales()
                    if prev_training:
                        model.train()
                    n_replaced = apply_static_scale_int8_decoder_ffn(model, ffn_scales)
                    sample = ", ".join(
                        f"{k.split('/')[-1]}={s:.4g}"
                        for k, s in list(ffn_scales.items())[:3]
                    )
                    print(
                        f"[INFO] static_scale_int8_decoder_ffn: replaced {n_replaced}/{len(ffn_targets)} "
                        f"(sample scales: {sample})"
                    )
        except Exception as e:
            import traceback
            print(f"[ERROR] static_scale_int8_decoder_ffn setup failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    if args.compile:
        if args.compile_encoder_only and args.compile_disable_encoder:
            print("[ERROR] --compile_encoder_only と --compile_disable_encoder は同時指定できません")
            sys.exit(1)
        if args.compile_disable_encoder and hasattr(model, "forward_encoder"):
            model.forward_encoder = MethodType(
                torch._dynamo.disable(model.forward_encoder.__func__),
                model,
            )
            print("[INFO] torch.compile: encoder disabled")
        if os.environ.get("TORCH_LOGS_OUT"):
            suppress_torch_compile_console_logs()
            print(f"[INFO] TORCH_LOGS_OUT={os.environ['TORCH_LOGS_OUT']}")
        if args.compile_cudagraphs == "on" and args.compile_mode == "max-autotune-no-cudagraphs":
            print("[ERROR] --compile_cudagraphs on は --compile_mode max-autotune-no-cudagraphs と同時指定できません")
            sys.exit(1)
        try:
            import torch._inductor.config as _inductor_config
            if hasattr(_inductor_config, "force_fuse_int_mm_with_mul"):
                _inductor_config.force_fuse_int_mm_with_mul = True
            # PyTorch 2.5.1 の Inductor triton_mm autotune は int8 mm で `tl.dot(allow_tf32=...)` 引数バグがあり
            # 34 件の triton.compiler.errors.CompilationError を起こす。Triton backend を外して ATEN (cuBLAS) のみ使う。
            if hasattr(_inductor_config, "max_autotune_gemm_backends"):
                _inductor_config.max_autotune_gemm_backends = "ATEN"
            # batch_linear / batch_linear_lhs は torchao 量子化テンソルで torch.stack を呼んで失敗する
            # ('aten.stack' not implemented for LinearActivationQuantizedTensor)
            if hasattr(_inductor_config, "pre_grad_fusion_options"):
                opts = dict(_inductor_config.pre_grad_fusion_options or {})
                opts.pop("batch_linear", None)
                opts.pop("batch_linear_lhs", None)
                _inductor_config.pre_grad_fusion_options = opts
        except Exception:
            pass
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.capture_scalar_outputs = True
            _dynamo.config.capture_dynamic_output_shape_ops = True
            _dynamo.config.force_parameter_static_shapes = False
            _dynamo.config.suppress_errors = True
        except Exception:
            pass
        compile_options = build_compile_options(args.compile_mode, args.compile_cudagraphs)
        compile_mode_arg, compile_options_arg = resolve_compile_call(args.compile_mode, compile_options)
        eager_quantized_linears = []
        if not args.compile_quantized_linear:
            eager_quantized_linears = wrap_quantized_linears_for_compile(model)
        try:
            compile_targets = []
            if args.compile_decoder_only and hasattr(model, "decoder"):
                model.decoder = torch.compile(
                    model.decoder,
                    mode=compile_mode_arg,
                    options=compile_options_arg,
                    dynamic=args.compile_dynamic,
                )
                compile_targets.append("decoder")
            if args.compile_encoder_only and hasattr(model, "forward_encoder"):
                model.forward_encoder = MethodType(
                    torch.compile(
                        model.forward_encoder.__func__,
                        mode=compile_mode_arg,
                        options=compile_options_arg,
                        dynamic=args.compile_dynamic,
                    ),
                    model,
                )
                compile_targets.append("encoder")
            if not compile_targets:
                model = torch.compile(
                    model,
                    mode=compile_mode_arg,
                    options=compile_options_arg,
                    dynamic=args.compile_dynamic,
                )
                compile_targets.append("model")
            compile_target_name = "+".join(compile_targets)
            head_compile_count = 0
            if args.compile_heads:
                for head_name in ['pose_head', 'shape_head', 'cam_head', 'conf_head', 'bbox_embed']:
                    head = getattr(model, head_name, None)
                    if head is None:
                        continue
                    if isinstance(head, nn.ModuleList):
                        for i in range(len(head)):
                            head[i] = torch.compile(
                                head[i],
                                mode=compile_mode_arg,
                                options=compile_options_arg,
                                dynamic=args.compile_dynamic,
                            )
                            head_compile_count += 1
                    else:
                        setattr(model, head_name, torch.compile(
                            head,
                            mode=compile_mode_arg,
                            options=compile_options_arg,
                            dynamic=args.compile_dynamic,
                        ))
                        head_compile_count += 1
                compile_target_name = compile_target_name + f"+heads({head_compile_count})"
            print(
                f"[INFO] torch.compile: target={compile_target_name} mode={args.compile_mode} "
                f"cudagraphs={args.compile_cudagraphs} dynamic={args.compile_dynamic} "
                f"compile_qlinear={args.compile_quantized_linear} "
                f"eager_qlinear_count={len(eager_quantized_linears)} "
                f"head_compile_count={head_compile_count}"
            )
        except Exception as e:
            print(f"[ERROR] torch.compile init failed: {e}")
            sys.exit(1)
    model.eval()
    if args.profile_forward:
        model._sat_profile_forward = True
        model._sat_profile_forward_limit = args.profile_forward_frames
        model._sat_profile_forward_count = 0
        model._sat_profile_encoder_blocks = args.profile_encoder_blocks
        print(f"[INFO] forward profile enabled: frames={args.profile_forward_frames}")
        if args.profile_encoder_blocks:
            print("[INFO] encoder block profile enabled")
    if not DTYPE_DEBUG_ONCE['model']:
        print(f"[DTYPE] model dtype: {next(model.parameters()).dtype}")
        DTYPE_DEBUG_ONCE['model'] = True

    if device.type != 'cuda' and args.batch_size > 1:
        print("[WARN] batch_size > 1 はCUDA実行のみ対応のため batch_size=1 にフォールバックします")
        args.batch_size = 1
    if args.batch_size > 1 and args.udp:
        print("[INFO] batch_size > 1 のため UDP送信を無効化します")
        args.udp = False
        args.udp_only = False

    print(f"デバイス: {device}")
    print(f"入力サイズ: {args.input_size}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"信頼度閾値: {args.conf_thresh}")
    if args.prewarm_frames > 0:
        print(
            f"空回し: {args.prewarm_frames} frame(s) "
            f"({'HELLO後' if args.prewarm_after_hello else 'HELLO前'})"
        )
    if args.no_render:
        print("メッシュ描画: OFF")
    if args.udp_only:
        print("UDP専用モード: ON (描画/表示を無効化)")
    if args.udp and not args.no_render:
        print("[WARN] 描画ONだとrender処理が重く、FPSが低下します。実運用は --udp_only または --no_render を推奨")
    if args.udp:
        print(f"UDP送信: ON (local:{args.local_port} -> ue:{args.ue_port})")
        if args.udp_trans_jump_cm > 0:
            print(
                f"外れ値抑制: ON (mode={args.udp_outlier_mode}, "
                f"trans_jump_cm={args.udp_trans_jump_cm:.1f}, max_skip={args.udp_outlier_max_skip})"
            )
        else:
            print("外れ値抑制: OFF")
        print(
            "ジャンプ例外: "
            f"conf>={args.udp_jump_conf_thresh:.2f}, "
            f"motion>={args.udp_jump_motion_cm:.1f}cm or "
            f"pose>={args.udp_jump_pose_motion:.4f}, "
            f"debounce={args.udp_jump_debounce}, "
            f"hold={args.udp_jump_hold_frames}, cooldown={args.udp_jump_cooldown}"
        )

    if args.prewarm_frames > 0 and not args.prewarm_after_hello:
        run_prewarm_frames(cap, model, args, device, dtype, total_frames)

    print("処理を開始...")

    pbar = tqdm(total=total_frames, desc="処理中")
    frame_idx = 0
    total_persons = 0
    all_times = {'preprocess': [], 'inference': [], 'postprocess': [], 'render': []}
    fps_queue = []
    metrics_fp = None
    metrics_writer = None
    if args.metrics_csv:
        metrics_dir = os.path.dirname(args.metrics_csv)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        metrics_fp = open(args.metrics_csv, 'w', newline='', encoding='utf-8')
        metrics_writer = csv.writer(metrics_fp)
        metrics_writer.writerow([
            'frame',
            'persons',
            'preprocess_ms',
            'inference_ms',
            'postprocess_ms',
            'render_ms',
            'total_ms',
            'fps_inst',
            'fps_avg_30',
            'udp_enabled',
        ])

    pred_frame_ids = []
    pred_image_paths = []
    pred_person_counts = []
    pred_joints2d = []
    pred_joints3d = []
    pred_verts = []
    pred_transl = []
    pred_bboxes = []
    pred_focal_length = []

    sock = None
    ue_host = None
    first_send_done = False
    time_connection_success = None
    prev_quat = None
    prev_trans = None
    outlier_skip_streak = 0
    jump_candidate_streak = 0
    jump_hold_left = 0
    jump_cooldown_left = 0
    first_frame_done = False
    processing_start_time = time.perf_counter()
    tracker_state = {
        'active': False,
        'transl': None,
        'query_id': None,
        'missed': 0,
    }

    if args.udp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", args.local_port))
        print(f"[WAIT] HELLO on 0.0.0.0:{args.local_port}")
        _, addr = sock.recvfrom(2048)
        ue_host = addr[0]
        print(f"[HELLO] from {addr} -> dst={(ue_host, args.ue_port)}")
        time_connection_success = time.perf_counter()
        if args.prewarm_frames > 0 and args.prewarm_after_hello:
            run_prewarm_frames(cap, model, args, device, dtype, total_frames)

    try:
        if args.batch_size == 1:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                if jump_cooldown_left > 0:
                    jump_cooldown_left -= 1
                if jump_hold_left > 0:
                    jump_hold_left -= 1

                result_frame, num_persons, times, udp_candidates, pred_record = process_frame(
                    model, frame, args.input_size, args.conf_thresh, device,
                    no_render=args.no_render, dtype=dtype
                )
                total_persons += num_persons

                if args.pred_npz:
                    pred_frame_ids.append(frame_idx)
                    pred_image_paths.append(f"{args.input}#frame_{frame_idx:06d}")
                    pred_person_counts.append(num_persons)
                    pred_joints2d.append(pred_record['joints2d'])
                    pred_joints3d.append(pred_record['joints3d'])
                    pred_verts.append(pred_record['verts'])
                    pred_transl.append(pred_record['transl'])
                    pred_bboxes.append(pred_record['bboxes'])
                    pred_focal_length.append(pred_record['focal_length'])

                for k, v in times.items():
                    all_times[k].append(v)

                if args.udp:
                    udp_data, tracker_state, track_reason = select_udp_target(
                        udp_candidates,
                        tracker_state,
                        args.udp_track_max_dist_cm,
                        args.udp_reinit_conf,
                    )

                    if udp_data is None:
                        outlier_skip_streak = 0
                        jump_candidate_streak = 0
                        tracker_state['missed'] += 1
                        if tracker_state['missed'] > args.udp_hold_frames:
                            tracker_state['active'] = False
                            tracker_state['transl'] = None
                            tracker_state['query_id'] = None
                        if frame_idx % 30 == 0:
                            print(f"UDP frame={frame_idx} skip ({track_reason}) persons={num_persons}")
                    else:
                        quaternions = smpl_pose_to_ue_quaternions(udp_data['pose'])
                        trans = transl_to_ue(udp_data['transl'])

                        if np.isfinite(quaternions).all() and np.isfinite(trans).all():
                            raw_trans_motion = 0.0 if prev_trans is None else float(np.linalg.norm(trans - prev_trans))
                            raw_pose_motion = 0.0 if prev_quat is None else float(np.mean(np.abs(quaternions - prev_quat)))

                            outlier_note = ""
                            send_trans = trans.copy()
                            skip_send = False

                            jump_armed = False
                            if prev_trans is not None and jump_cooldown_left == 0:
                                jump_cond = (
                                    udp_data['conf'] >= args.udp_jump_conf_thresh and
                                    (
                                        raw_trans_motion >= args.udp_jump_motion_cm or
                                        raw_pose_motion >= args.udp_jump_pose_motion
                                    )
                                )
                                if jump_cond:
                                    jump_candidate_streak += 1
                                    if jump_candidate_streak >= max(1, args.udp_jump_debounce):
                                        jump_hold_left = max(jump_hold_left, args.udp_jump_hold_frames)
                                        jump_candidate_streak = 0
                                else:
                                    jump_candidate_streak = 0
                            else:
                                jump_candidate_streak = 0

                            if jump_hold_left > 0:
                                jump_armed = True
                                outlier_skip_streak = 0
                                outlier_note = f" jump_bypass(hold={jump_hold_left})"

                            if (not jump_armed) and prev_trans is not None and args.udp_trans_jump_cm > 0 and raw_trans_motion > args.udp_trans_jump_cm:
                                if args.udp_outlier_mode == 'skip':
                                    outlier_skip_streak += 1
                                    if outlier_skip_streak <= args.udp_outlier_max_skip:
                                        skip_send = True
                                        outlier_note = (
                                            f"trans_outlier {raw_trans_motion:.2f}>{args.udp_trans_jump_cm:.2f}cm "
                                            f"(skip {outlier_skip_streak}/{args.udp_outlier_max_skip})"
                                        )
                                    else:
                                        outlier_note = (
                                            f" outlier=rebase({raw_trans_motion:.2f}cm "
                                            f"after {outlier_skip_streak} skips)"
                                        )
                                        outlier_skip_streak = 0
                                else:
                                    delta = trans - prev_trans
                                    scale = args.udp_trans_jump_cm / (raw_trans_motion + 1e-9)
                                    send_trans = prev_trans + delta * scale
                                    outlier_note = (
                                        f" outlier=clamp({raw_trans_motion:.2f}"
                                        f"->{args.udp_trans_jump_cm:.2f}cm)"
                                    )
                                    outlier_skip_streak = 0
                            else:
                                outlier_skip_streak = 0

                            if skip_send:
                                if frame_idx % 30 == 0:
                                    print(f"UDP frame={frame_idx} skip ({outlier_note}) persons={num_persons}")
                            else:
                                pose_str = ",".join(f"{v:.4f}" for v in quaternions.flatten())
                                trans_str = ",".join(f"{v:.4f}" for v in send_trans.flatten())
                                payload = f"{pose_str}|{trans_str}\n".encode("ascii")
                                sent = sock.sendto(payload, (ue_host, args.ue_port))

                                if not first_send_done:
                                    startup_lag = time.perf_counter() - time_connection_success
                                    print(f"[INFO] Startup Lag: {startup_lag:.4f} seconds")
                                    first_send_done = True

                                if prev_quat is None:
                                    pose_motion = 0.0
                                    trans_motion = 0.0
                                else:
                                    pose_motion = raw_pose_motion
                                    trans_motion = float(np.linalg.norm(send_trans - prev_trans))

                                prev_quat = quaternions.copy()
                                prev_trans = send_trans.copy()

                                if jump_armed and jump_hold_left == 1:
                                    jump_cooldown_left = max(0, args.udp_jump_cooldown)

                                if frame_idx % 30 == 0:
                                    print(
                                        f"UDP frame={frame_idx} conf={udp_data['conf']:.3f} "
                                        f"persons={udp_data['num_persons']} query={udp_data['query_id']} "
                                        f"reason={track_reason} pose_motion={pose_motion:.5f} "
                                        f"trans_motion={trans_motion:.2f}{outlier_note} sent={sent}"
                                    )
                        elif frame_idx % 30 == 0:
                            print(f"UDP frame={frame_idx} skip (invalid payload) persons={num_persons}")

                elapsed = time.time() - start_time
                if not first_frame_done:
                    total_startup_time = time.perf_counter() - processing_start_time
                    print(
                        f"[INFO] First frame completed: frame={frame_idx} "
                        f"total={total_startup_time:.4f}s "
                        f"inference={times['inference'] * 1000.0:.1f}ms"
                    )
                    first_frame_done = True
                current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                fps_queue.append(current_fps)
                if len(fps_queue) > 30:
                    fps_queue.pop(0)
                avg_fps_30 = float(np.mean(fps_queue)) if fps_queue else 0.0
                if metrics_writer is not None:
                    metrics_writer.writerow([
                        frame_idx,
                        num_persons,
                        f'{times["preprocess"] * 1000.0:.4f}',
                        f'{times["inference"] * 1000.0:.4f}',
                        f'{times["postprocess"] * 1000.0:.4f}',
                        f'{times["render"] * 1000.0:.4f}' if 'render' in times else '',
                        f'{elapsed * 1000.0:.4f}',
                        f'{current_fps:.6f}',
                        f'{avg_fps_30:.6f}',
                        int(args.udp),
                    ])
                    metrics_fp.flush()

                if writer is not None:
                    writer.write(result_frame)

                if args.show_fps and args.display:
                    cv2.putText(result_frame, f"FPS: {current_fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Persons: {num_persons}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    y_offset = 110
                    cv2.putText(result_frame, f"Preprocess: {times['preprocess']*1000:.1f}ms", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 30
                    cv2.putText(result_frame, f"Inference: {times['inference']*1000:.1f}ms", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 30
                    cv2.putText(result_frame, f"Postprocess: {times['postprocess']*1000:.1f}ms", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    if 'render' in times:
                        y_offset += 30
                        cv2.putText(result_frame, f"Render: {times['render']*1000:.1f}ms", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if args.display:
                    cv2.imshow('SAT-HMR Video Demo', result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n中断されました")
                        break
                elif frame_idx % 30 == 0:
                    render_ms = f"{times['render']*1000:.0f}ms" if 'render' in times else 'skip'
                    print(
                        f"Frame: {frame_idx}/{total_frames} | FPS: {current_fps:.1f} | Persons: {num_persons} | "
                        f"Preprocess: {times['preprocess']*1000:.0f}ms | "
                        f"Inference: {times['inference']*1000:.0f}ms | "
                        f"Postprocess: {times['postprocess']*1000:.0f}ms | "
                        f"Render: {render_ms}"
                    )

                frame_idx += 1
                pbar.update(1)
                pbar.set_postfix({'persons': num_persons, 'fps': f'{current_fps:.1f}'})
        else:
            stop_requested = False
            while not stop_requested:
                batch_frames = []
                while len(batch_frames) < args.batch_size:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)

                if not batch_frames:
                    break

                batch_results = process_frames_batch(
                    model=model,
                    frames=batch_frames,
                    input_size=args.input_size,
                    conf_thresh=args.conf_thresh,
                    device=device,
                    no_render=args.no_render,
                    dtype=dtype,
                )

                for result_frame, num_persons, times, _udp_candidates, pred_record in batch_results:
                    total_persons += num_persons

                    if args.pred_npz:
                        pred_frame_ids.append(frame_idx)
                        pred_image_paths.append(f"{args.input}#frame_{frame_idx:06d}")
                        pred_person_counts.append(num_persons)
                        pred_joints2d.append(pred_record['joints2d'])
                        pred_joints3d.append(pred_record['joints3d'])
                        pred_verts.append(pred_record['verts'])
                        pred_transl.append(pred_record['transl'])
                        pred_bboxes.append(pred_record['bboxes'])
                        pred_focal_length.append(pred_record['focal_length'])

                    for k, v in times.items():
                        all_times[k].append(v)

                    elapsed = (
                        times.get('preprocess', 0.0)
                        + times.get('inference', 0.0)
                        + times.get('postprocess', 0.0)
                        + times.get('render', 0.0)
                    )
                    if not first_frame_done:
                        total_startup_time = time.perf_counter() - processing_start_time
                        print(
                            f"[INFO] First frame completed: frame={frame_idx} "
                            f"total={total_startup_time:.4f}s "
                            f"inference={times['inference'] * 1000.0:.1f}ms"
                        )
                        first_frame_done = True
                    current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                    fps_queue.append(current_fps)
                    if len(fps_queue) > 30:
                        fps_queue.pop(0)
                    avg_fps_30 = float(np.mean(fps_queue)) if fps_queue else 0.0

                    if metrics_writer is not None:
                        metrics_writer.writerow([
                            frame_idx,
                            num_persons,
                            f'{times["preprocess"] * 1000.0:.4f}',
                            f'{times["inference"] * 1000.0:.4f}',
                            f'{times["postprocess"] * 1000.0:.4f}',
                            f'{times["render"] * 1000.0:.4f}' if 'render' in times else '',
                            f'{elapsed * 1000.0:.4f}',
                            f'{current_fps:.6f}',
                            f'{avg_fps_30:.6f}',
                            0,
                        ])
                        metrics_fp.flush()

                    if writer is not None:
                        writer.write(result_frame)

                    if args.show_fps and args.display:
                        cv2.putText(result_frame, f"FPS: {current_fps:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(result_frame, f"Persons: {num_persons}", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        y_offset = 110
                        cv2.putText(result_frame, f"Preprocess: {times['preprocess']*1000:.1f}ms", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        y_offset += 30
                        cv2.putText(result_frame, f"Inference: {times['inference']*1000:.1f}ms", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        y_offset += 30
                        cv2.putText(result_frame, f"Postprocess: {times['postprocess']*1000:.1f}ms", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        if 'render' in times:
                            y_offset += 30
                            cv2.putText(result_frame, f"Render: {times['render']*1000:.1f}ms", (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    if args.display:
                        cv2.imshow('SAT-HMR Video Demo', result_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            print("\n中断されました")
                            stop_requested = True
                            break
                    elif frame_idx % 30 == 0:
                        render_ms = f"{times['render']*1000:.0f}ms" if 'render' in times else 'skip'
                        print(
                            f"Frame: {frame_idx}/{total_frames} | FPS: {current_fps:.1f} | Persons: {num_persons} | "
                            f"Preprocess: {times['preprocess']*1000:.0f}ms | "
                            f"Inference: {times['inference']*1000:.0f}ms | "
                            f"Postprocess: {times['postprocess']*1000:.0f}ms | "
                            f"Render: {render_ms}"
                        )

                    frame_idx += 1
                    pbar.update(1)
                    pbar.set_postfix({'persons': num_persons, 'fps': f'{current_fps:.1f}'})

    except KeyboardInterrupt:
        print("\n中断されました")

    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
        if sock is not None:
            sock.close()
        if metrics_fp is not None:
            metrics_fp.close()
        cv2.destroyAllWindows()

    if args.pred_npz:
        pred_dir = os.path.dirname(args.pred_npz)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        np.savez_compressed(
            args.pred_npz,
            frame_ids=np.asarray(pred_frame_ids, dtype=np.int64),
            image_paths=np.asarray(pred_image_paths, dtype=object),
            person_counts=np.asarray(pred_person_counts, dtype=np.int32),
            joints2d=np.asarray(pred_joints2d, dtype=object),
            joints3d=np.asarray(pred_joints3d, dtype=object),
            verts=np.asarray(pred_verts, dtype=object),
            transl=np.asarray(pred_transl, dtype=object),
            bboxes=np.asarray(pred_bboxes, dtype=object),
            focal_length=np.asarray(pred_focal_length, dtype=object),
            failed_frames=np.int32(0),
        )

    print(f"\n処理完了!")
    print(f"処理フレーム数: {frame_idx}")
    print(f"平均検出人数: {total_persons / max(frame_idx, 1):.2f}")

    warmup_n = max(0, getattr(args, 'warmup_frames', 0))
    if warmup_n > 0:
        print(f"\n--- 処理時間 (平均, warmup={warmup_n} フレーム除外) ---")
    else:
        print(f"\n--- 処理時間 (平均) ---")
    total_time = 0
    for k, v in all_times.items():
        if len(v) > 0:
            sample = v[warmup_n:] if warmup_n < len(v) else v
            if len(sample) == 0:
                continue
            avg = np.mean(sample) * 1000
            total_time += avg
            print(f"  {k}: {avg:.1f} ms")
    print(f"  合計: {total_time:.1f} ms ({1000/max(total_time, 1e-9):.1f} FPS)")
    
    if not args.no_save:
        print(f"\n出力: {args.output}")
    if args.metrics_csv:
        print(f"測定CSV: {args.metrics_csv}")
    if args.pred_npz:
        print(f"予測NPZ: {args.pred_npz}")
    print(f"実行ログ: {_RUN_LOG_PATH}")


if __name__ == '__main__':
    main()
