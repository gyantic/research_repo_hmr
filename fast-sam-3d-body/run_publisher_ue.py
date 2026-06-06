"""Real-time SMPL publisher with Unreal Engine UDP/TCP streaming.

run_publisher.py をベースに、ZMQ 配信に加えて UE への UDP/TCP 送信を行う。

使い方 (UDP):
    python run_publisher_ue.py \\
        --source camera \\
        --smpl-model-path <smpl.pkl> \\
        --nn-model-dir <model_dir> \\
        --mhr2smpl-mapping-path <mapping.npz> \\
        --ue udp \\
        --ue-port 62004 \\
        --local-port 5555

使い方 (TCP):
    python run_publisher_ue.py \\
        --source camera \\
        --smpl-model-path <smpl.pkl> \\
        --nn-model-dir <model_dir> \\
        --mhr2smpl-mapping-path <mapping.npz> \\
        --ue tcp \\
        --local-port 5555
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")
os.environ.setdefault("SAM3D_QUIET", "1")

# Performance settings (from run_demo.sh)
os.environ.setdefault("USE_TRT_BACKBONE", "1")
os.environ.setdefault("TRT_BACKBONE_PATH", "./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_int8.engine")
os.environ.setdefault("GPU_HAND_PREP", "1")
os.environ.setdefault("LAYER_DTYPE", "fp32")
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")
os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")
os.environ.setdefault("IMG_SIZE", "512")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")
os.environ.setdefault("FOV_TRT", "1")
os.environ.setdefault("FOV_TRT_PATH", "./checkpoints/moge_trt/moge_dinov2_encoder_int8.engine")
os.environ.setdefault("FOV_FAST", "1")
os.environ.setdefault("FOV_MODEL", "s")
os.environ.setdefault("FOV_LEVEL", "0")
os.environ.setdefault("BODY_INTERM_PRED_LAYERS", "0,1,2")
os.environ.setdefault("HAND_INTERM_PRED_LAYERS", "0,1")
os.environ.setdefault("MHR_USE_CUDA_GRAPH", "1")
os.environ.setdefault("USE_COMPILE", "1")
os.environ.setdefault("USE_COMPILE_BACKBONE", "1")
os.environ.setdefault("DECODER_COMPILE", "1")
os.environ.setdefault("COMPILE_MODE", "reduce-overhead")
os.environ.setdefault("COMPILE_WARMUP_BATCH_SIZES", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/triton"))

import json
import queue
import signal
import cv2
import numpy as np
import torch
from loguru import logger

_LOG_LEVEL = "INFO"
logger.remove()
logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)

from scipy.spatial.transform import Rotation

from mocap.core.gravity_alignment import build_camera_to_world_rotation
from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner
from mocap.realtime.interpolator import PoseInterpolator
from mocap.realtime.publisher import ZMQPublisher
from mocap.realtime.ue_sender import UESender
from mocap.core.setup_estimator import build_default_estimator
from mocap.utils.pose_protocol import prepare_publish_pose
from mocap.utils.video_source import create_video_source


FOV_MODEL_SIZE = "s"
FOV_RESOLUTION_LEVEL = 0
FOV_FIXED_SIZE = 512
FOV_FAST_MODE = True
YOLO_MODEL_PATH = "checkpoints/yolo/yolo11m-pose.engine"


class RealtimePublisherUE:
    """run_publisher.py の RealtimePublisher に UE 送信機能を追加したクラス。

    ZMQ ストリーム (既存) と UE UDP/TCP 送信 (追加) を並行して行う。
    """

    def __init__(
        self,
        video_source,
        publish_hz,
        interpolate_lag_ms,
        smpl_model_path,
        multiview_model_dir,
        mhr2smpl_mapping_path,
        mhr_mesh_path=None,
        smoother_dir=None,
        addr="tcp://*:5556",
        image_size=512,
        yolo_model_path=YOLO_MODEL_PATH,
        record=False,
        record_dir="output/records",
        min_person_confidence=0.75,
        ue_sender=None,
        metrics_csv=None,
        summary_csv=None,
    ):
        logger.info("Initializing Realtime Publisher (UE)...")
        self.min_person_confidence = min_person_confidence

        logger.info("Loading SAM 3D model...")
        self.estimator = build_default_estimator(
            image_size=image_size,
            yolo_model_path=yolo_model_path,
            fov_model_size=FOV_MODEL_SIZE,
            fov_resolution_level=FOV_RESOLUTION_LEVEL,
            fov_fixed_size=FOV_FIXED_SIZE,
            fov_fast_mode=FOV_FAST_MODE,
        )

        self.video_source = video_source

        logger.info("Warming up model...")
        self._warmup()

        cam_intrinsics_np = self.video_source.get_camera_intrinsics()
        if cam_intrinsics_np is not None:
            self.cam_intrinsics = torch.from_numpy(
                np.asarray(cam_intrinsics_np, dtype=np.float32)
            )
            logger.info(
                f"Using camera intrinsics: fx={cam_intrinsics_np[0,0,0]:.2f}, fy={cam_intrinsics_np[0,1,1]:.2f}"
            )
        else:
            self.cam_intrinsics = None
            logger.warning("No camera intrinsics provided, will use FOV estimator")

        self.gravity_direction = self.video_source.get_gravity_direction()
        self.R_world_cam = build_camera_to_world_rotation(self.gravity_direction)
        R_zup_adjustment = np.array(
            [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64
        )
        self.R_world_cam = R_zup_adjustment @ self.R_world_cam

        self.record = record
        self.record_dir = record_dir
        if self.record:
            os.makedirs(self.record_dir, exist_ok=True)
            session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir = os.path.join(self.record_dir, session_id)
            os.makedirs(self.session_dir, exist_ok=True)
            self.video_out_path = os.path.join(self.session_dir, "raw_video.mp4")
            self.smpl_out_path = os.path.join(self.session_dir, "smpl_data.npz")
            logger.info(f"Recording enabled. Saving to: {self.session_dir}")

            intrinsics_data = {
                "width": 848,
                "height": 480,
                "gravity": self.gravity_direction.tolist(),
            }
            if cam_intrinsics_np is not None:
                intrinsics_data["fx"] = float(cam_intrinsics_np[0, 0, 0])
                intrinsics_data["fy"] = float(cam_intrinsics_np[0, 1, 1])
                intrinsics_data["cx"] = float(cam_intrinsics_np[0, 0, 2])
                intrinsics_data["cy"] = float(cam_intrinsics_np[0, 1, 2])
                intrinsics_data["camera_matrix"] = cam_intrinsics_np[0].tolist()

            frame_size = self.video_source.get_frame_size()
            if frame_size is not None:
                intrinsics_data["width"] = frame_size[0]
                intrinsics_data["height"] = frame_size[1]

            with open(os.path.join(self.session_dir, "camera.json"), "w") as f:
                json.dump(intrinsics_data, f, indent=2)

            self.video_queue = queue.Queue(maxsize=300)
            self.smpl_queue = queue.Queue(maxsize=300)
            self.smpl_data_list = []

        logger.info(f"Loading multiview fusion runner model={smpl_model_path}...")
        self.fusion_runner = MultiViewFusionRunner(
            smpl_model_path=smpl_model_path,
            model_dir=multiview_model_dir,
            mapping_path=mhr2smpl_mapping_path,
            mhr_mesh_path=mhr_mesh_path,
            smoother_dir=smoother_dir,
        )

        self.publish_hz = publish_hz
        self.publish_dt = 1.0 / publish_hz
        self.interpolate_lag_s = interpolate_lag_ms / 1000.0

        self.interpolator = PoseInterpolator()
        self.publisher = ZMQPublisher(addr)

        # UE sender (None の場合は UE 送信を行わない)
        self.ue_sender = ue_sender
        self.metrics_csv = metrics_csv
        self.summary_csv = summary_csv

        # UE 送信 FPS 計測
        self._ue_send_times = deque(maxlen=60)
        self._ue_send_lock = threading.Lock()

        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        self.running = False
        self.video_ended = False
        self._final_stats_logged = False
        self._publish_enabled = True
        self._closed = False

        self._capture_wall_base = None
        self._capture_ts_base = None
        self._pose_clock_lock = threading.Lock()
        self._latest_pose_source_ts = None
        self._latest_pose_perf_ts = None

        self.capture_thread = None
        self.inference_thread = None
        self.publish_thread = None
        self.recording_thread = None

        self.first_capture_ts = None
        self.first_infer_ts = None

        self.stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "infer_count": 0,
            "inference_times": deque(maxlen=100),
            "infer_total_time_s": 0.0,
            "convert_times": deque(maxlen=100),
            "convert_total_time_s": 0.0,
            "publish_count": 0,
            "publish_intervals": deque(maxlen=500),
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
            "backbone_times_ms": deque(maxlen=100),
            "decoders_times_ms": deque(maxlen=100),
            "detection_times_ms": deque(maxlen=100),
            "fov_times_ms": deque(maxlen=100),
            "frame_stats": [],
            "ue_send_count": 0,
        }

        self._live_log_interval_s = 2.0
        self._live_last_log_perf = time.perf_counter()
        self._live_prev_stats = {
            "capture_count": 0,
            "dropped_capture_count": 0,
            "infer_count": 0,
            "infer_total_time_s": 0.0,
            "convert_total_time_s": 0.0,
            "publish_count": 0,
            "publish_interpolated_count": 0,
            "publish_fallback_count": 0,
        }

        logger.success("Publisher (UE) ready")

    def _warmup(self):
        frame_size = self.video_source.get_frame_size()
        if frame_size is None:
            width, height = 640, 480
        else:
            width, height = frame_size

        dummy_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        warmup_bbox = np.array(
            [[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32
        )
        for _ in range(2):
            _ = self.estimator.process_one_image(
                dummy_img,
                bboxes=warmup_bbox,
                hand_box_source="body_decoder",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _capture_loop(self):
        while self.running:
            try:
                frame, frame_timestamp = self.video_source.get_frame()
            except Exception as exc:
                if self.running:
                    logger.warning(f"Capture loop stopped due to source error: {exc}")
                self.video_ended = True
                self._frame_event.set()
                break

            if frame is None:
                self.video_ended = True
                self._frame_event.set()
                break

            if frame_timestamp is None:
                continue

            if self.first_capture_ts is None:
                self.first_capture_ts = frame_timestamp

            if self._capture_wall_base is None:
                self._capture_wall_base = time.perf_counter()
                self._capture_ts_base = frame_timestamp
            else:
                target_wall = self._capture_wall_base + (
                    frame_timestamp - self._capture_ts_base
                )
                now_wall = time.perf_counter()
                delay = target_wall - now_wall
                if delay > 0:
                    time.sleep(delay)

            self.stats["capture_count"] += 1

            if self.record:
                try:
                    self.video_queue.put_nowait((frame_timestamp, frame))
                except queue.Full:
                    logger.warning(
                        "Video recording queue full, dropping frame for recording"
                    )

            with self._latest_frame_lock:
                if self._latest_frame is not None:
                    self.stats["dropped_capture_count"] += 1
                self._latest_frame = (frame, frame_timestamp)
            self._frame_event.set()

    def _inference_loop(self):
        while self.running:
            self._frame_event.wait(timeout=0.05)
            self._frame_event.clear()

            with self._latest_frame_lock:
                item = self._latest_frame
                self._latest_frame = None

            if item is None:
                if self.video_ended:
                    break
                continue

            frame, frame_timestamp = item

            if self.first_infer_ts is None:
                self.first_infer_ts = frame_timestamp

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t_infer_start = time.perf_counter()
            outputs = self.estimator.process_one_image(
                frame_rgb,
                cam_int=self.cam_intrinsics,
                hand_box_source="yolo_pose",
            )
            infer_dt = time.perf_counter() - t_infer_start
            infer_ms = infer_dt * 1000.0

            self.stats["inference_times"].append(infer_dt)
            self.stats["infer_total_time_s"] += infer_dt
            infer_index = self.stats["infer_count"]
            self.stats["infer_count"] += 1
            t = self.estimator.last_timings or {}
            if t:
                self.stats["backbone_times_ms"].append(t.get("backbone_ms", 0))
                self.stats["decoders_times_ms"].append(t.get("decoders_ms", 0))
                self.stats["detection_times_ms"].append(t.get("detection_ms", 0))
                self.stats["fov_times_ms"].append(t.get("fov_ms", 0))

            num_persons = len(outputs)
            frame_stat = {
                "infer_index": infer_index,
                "source_timestamp": frame_timestamp,
                "persons": num_persons,
                "infer_ms": infer_ms,
                "backbone_ms": float(t.get("backbone_ms", np.nan)),
                "decoders_ms": float(t.get("decoders_ms", np.nan)),
                "detection_ms": float(t.get("detection_ms", np.nan)),
                "fov_ms": float(t.get("fov_ms", np.nan)),
                "convert_ms": np.nan,
                "send_ms": np.nan,
                "total_ms": np.nan,
                "ue_sent": 0,
                "status": "inference_only",
            }
            if num_persons != 1:
                if self._publish_enabled:
                    logger.warning(
                        "Pose publishing paused (not exactly 1 confident person detected)."
                    )
                    self._publish_enabled = False
                frame_stat["status"] = f"person_count_{num_persons}"
                self.stats["frame_stats"].append(frame_stat)
                continue

            if not self._publish_enabled:
                logger.info("Exactly 1 person confirmed; resuming pose publishing.")
                self._publish_enabled = True

            t0 = time.perf_counter()
            out = outputs[0]
            required = ("pred_vertices", "pred_cam_t", "global_rot")
            missing = [k for k in required if k not in out]
            if missing:
                logger.warning(f"Skipping frame missing keys: missing={missing}")
                frame_stat["status"] = "missing_" + ",".join(missing)
                self.stats["frame_stats"].append(frame_stat)
                continue

            pred_vertices = np.asarray(out["pred_vertices"], dtype=np.float32)
            pred_cam_t = np.asarray(out["pred_cam_t"], dtype=np.float32)
            global_rot = np.asarray(out["global_rot"], dtype=np.float64).reshape(3)

            body_quat_xyzw = self._compute_body_quat(global_rot)
            smpl_pose, canonical_joints, _betas, _weights = self.fusion_runner.infer(
                [(pred_vertices, pred_cam_t)]
            )
            body_quat, smpl_joints, smpl_pose_pub = self._prepare_publish_pose(
                body_quat_xyzw, canonical_joints, smpl_pose
            )
            convert_dt = time.perf_counter() - t0
            self.stats["convert_times"].append(convert_dt)
            self.stats["convert_total_time_s"] += convert_dt
            convert_ms = convert_dt * 1000.0
            frame_stat["convert_ms"] = convert_ms

            # ZMQ interpolator に root_trans (pred_cam_t) も渡す
            self.interpolator.add_pose(
                frame_timestamp,
                body_quat,
                smpl_joints,
                smpl_pose_pub,
                root_trans=pred_cam_t,
            )

            # UE 即時送信 (推論完了直後)
            if self.ue_sender is not None and self.ue_sender.is_connected:
                t_send = time.perf_counter()
                self.ue_sender.send(
                    global_rot,
                    np.asarray(smpl_pose, dtype=np.float32).reshape(21, 3),
                    pred_cam_t,
                )
                frame_stat["send_ms"] = (time.perf_counter() - t_send) * 1000.0
                frame_stat["ue_sent"] = 1
                frame_stat["status"] = "sent"
                self.stats["ue_send_count"] += 1
                with self._ue_send_lock:
                    self._ue_send_times.append(time.perf_counter())
            else:
                frame_stat["status"] = "converted_not_sent"

            frame_stat["total_ms"] = (
                infer_ms
                + frame_stat["convert_ms"]
                + (0.0 if np.isnan(frame_stat["send_ms"]) else frame_stat["send_ms"])
            )
            self.stats["frame_stats"].append(frame_stat)

            if self.record:
                try:
                    self.smpl_queue.put_nowait(
                        (frame_timestamp, body_quat, smpl_joints, smpl_pose_pub)
                    )
                except queue.Full:
                    logger.warning(
                        "SMPL recording queue full, dropping pose for recording"
                    )

            with self._pose_clock_lock:
                self._latest_pose_source_ts = frame_timestamp
                self._latest_pose_perf_ts = time.perf_counter()

    def _compute_body_quat(self, global_rot):
        rot = Rotation.from_euler("ZYX", global_rot)
        x180 = Rotation.from_euler("x", 180.0, degrees=True)
        return (x180 * rot).as_quat().astype(np.float64)

    def _prepare_publish_pose(self, body_quat_xyzw, canonical_joints, smpl_pose):
        return prepare_publish_pose(
            body_quat_xyzw,
            canonical_joints,
            smpl_pose,
            self.R_world_cam,
        )

    def _maybe_log_live_stats(self, now_perf):
        elapsed = now_perf - self._live_last_log_perf
        if elapsed < self._live_log_interval_s:
            return

        curr = {
            "capture_count": self.stats["capture_count"],
            "dropped_capture_count": self.stats["dropped_capture_count"],
            "infer_count": self.stats["infer_count"],
            "infer_total_time_s": self.stats["infer_total_time_s"],
            "convert_total_time_s": self.stats["convert_total_time_s"],
            "publish_count": self.stats["publish_count"],
            "publish_interpolated_count": self.stats["publish_interpolated_count"],
            "publish_fallback_count": self.stats["publish_fallback_count"],
        }
        prev = self._live_prev_stats

        d_capture = curr["capture_count"] - prev["capture_count"]
        d_drop = curr["dropped_capture_count"] - prev["dropped_capture_count"]
        d_infer = curr["infer_count"] - prev["infer_count"]
        d_infer_time = curr["infer_total_time_s"] - prev["infer_total_time_s"]
        d_convert_time = curr["convert_total_time_s"] - prev["convert_total_time_s"]
        d_publish = curr["publish_count"] - prev["publish_count"]
        d_interp = curr["publish_interpolated_count"] - prev["publish_interpolated_count"]
        d_fallback = curr["publish_fallback_count"] - prev["publish_fallback_count"]

        infer_fps = d_infer / elapsed
        infer_ms = (d_infer_time / d_infer * 1000.0) if d_infer > 0 else float("nan")
        convert_ms = (d_convert_time / d_infer * 1000.0) if d_infer > 0 else float("nan")
        publish_hz = d_publish / elapsed
        fallback_pct = (d_fallback / d_publish * 100.0) if d_publish > 0 else 0.0

        infer_ms_str = f"{infer_ms:.1f}" if np.isfinite(infer_ms) else "n/a"
        convert_ms_str = f"{convert_ms:.1f}" if np.isfinite(convert_ms) else "n/a"

        # Per-component breakdown from last inference
        t = self.estimator.last_timings
        breakdown = (
            f" [det={t.get('detection_ms', 0):.0f} bb={t.get('backbone_ms', 0):.0f} dec={t.get('decoders_ms', 0):.0f} model={t.get('model_ms', 0):.0f}]ms"
            if t else ""
        )

        logger.info(
            "Live: "
            f"capture={d_capture/elapsed:.1f}fps drop+={d_drop}, "
            f"infer throughput={infer_fps:.1f}fps model={infer_ms_str}ms{breakdown}, "
            f"convert={convert_ms_str}ms, "
            f"publish={publish_hz:.1f}Hz, "
            f"interp+={d_interp}, fallback+={d_fallback} ({fallback_pct:.1f}%)"
        )

        if self.ue_sender is not None:
            with self._ue_send_lock:
                times = list(self._ue_send_times)
            if len(times) >= 2:
                ue_fps = (len(times) - 1) / (times[-1] - times[0])
            else:
                ue_fps = 0.0
            logger.info(
                f"UE send: {ue_fps:.1f}fps  infer={infer_fps:.1f}fps ({infer_ms_str}ms)"
            )

        self._live_prev_stats = curr
        self._live_last_log_perf = now_perf

    def _recording_loop(self):
        video_writer = None
        fps = self.video_source.fps
        while self.running:
            try:
                frame_ts, frame = self.video_queue.get(timeout=0.05)
                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    video_writer = cv2.VideoWriter(
                        self.video_out_path, fourcc, fps, (w, h)
                    )
                if video_writer is not None:
                    video_writer.write(frame)
            except queue.Empty:
                pass

            try:
                while True:
                    smpl_ts, body_quat, smpl_joints, smpl_pose = (
                        self.smpl_queue.get_nowait()
                    )
                    self.smpl_data_list.append(
                        {
                            "timestamp": smpl_ts,
                            "body_quat": body_quat,
                            "smpl_joints": smpl_joints,
                            "smpl_pose": smpl_pose,
                        }
                    )
            except queue.Empty:
                pass

        if self.record:
            logger.info("Flushing recording queues to disk. Please wait...")
            while not self.video_queue.empty():
                try:
                    frame_ts, frame = self.video_queue.get_nowait()
                    if video_writer is not None:
                        video_writer.write(frame)
                except queue.Empty:
                    break

            while not self.smpl_queue.empty():
                try:
                    smpl_ts, body_quat, smpl_joints, smpl_pose = (
                        self.smpl_queue.get_nowait()
                    )
                    self.smpl_data_list.append(
                        {
                            "timestamp": smpl_ts,
                            "body_quat": body_quat,
                            "smpl_joints": smpl_joints,
                            "smpl_pose": smpl_pose,
                        }
                    )
                except queue.Empty:
                    break

            if video_writer is not None:
                video_writer.release()
                logger.info(f"Finished writing video to {self.video_out_path}")

            if self.smpl_data_list:
                timestamps = np.array([d["timestamp"] for d in self.smpl_data_list])
                body_quats = np.array([d["body_quat"] for d in self.smpl_data_list])
                smpl_joints = np.array([d["smpl_joints"] for d in self.smpl_data_list])
                smpl_poses = np.array([d["smpl_pose"] for d in self.smpl_data_list])
                np.savez(
                    self.smpl_out_path,
                    timestamps=timestamps,
                    body_quats=body_quats,
                    smpl_joints=smpl_joints,
                    smpl_poses=smpl_poses,
                )
                logger.info(f"Finished writing SMPL data to {self.smpl_out_path}")

    def _publish_loop(self):
        last_publish = None
        next_publish = time.perf_counter()

        while self.running:
            now_perf = time.perf_counter()
            wait_time = next_publish - now_perf
            if wait_time > 0:
                time.sleep(min(wait_time, 0.0015))
                continue

            wall_now = time.time()
            with self._pose_clock_lock:
                latest_pose_source_ts = self._latest_pose_source_ts
                latest_pose_perf_ts = self._latest_pose_perf_ts

            if latest_pose_source_ts is None or latest_pose_perf_ts is None:
                query_ts = wall_now - self.interpolate_lag_s
            else:
                source_now_est = latest_pose_source_ts + (
                    now_perf - latest_pose_perf_ts
                )
                query_ts = source_now_est - self.interpolate_lag_s

            result = self.interpolator.interpolate(query_ts)
            used_fallback = False
            if result is None:
                latest = self.interpolator.get_latest_pose()
                if latest is not None:
                    # get_latest_pose は (body_quat, smpl_joints, smpl_pose) の 3 タプルを返す
                    result = (*latest, None)
                    used_fallback = True

            if result is not None and self._publish_enabled:
                body_quat, smpl_joints, smpl_pose, _root_trans = result

                # ZMQ 配信 (既存プロトコル)
                self.publisher.publish(body_quat, smpl_joints, smpl_pose)
                self.stats["publish_count"] += 1
                if used_fallback:
                    self.stats["publish_fallback_count"] += 1
                else:
                    self.stats["publish_interpolated_count"] += 1
                if last_publish is not None:
                    self.stats["publish_intervals"].append(now_perf - last_publish)
                last_publish = now_perf


            next_publish += self.publish_dt
            if next_publish < now_perf - self.publish_dt:
                missed = int((now_perf - next_publish) / self.publish_dt) + 1
                next_publish += missed * self.publish_dt

            if self.video_ended and self.stats["infer_count"] > 0:
                if self.stats["publish_count"] > max(10, int(self.publish_hz * 0.4)):
                    break

            self._maybe_log_live_stats(now_perf)

    def _log_final_stats(self):
        if self._final_stats_logged:
            return

        inf_msg = "Inference n/a"
        if self.stats["inference_times"]:
            inf_mean = np.mean(self.stats["inference_times"]) * 1000
            inf_fps = 1.0 / np.mean(self.stats["inference_times"])
            inf_msg = f"Inference {inf_mean:.1f}ms ({inf_fps:.1f}fps)"
            if self.stats["convert_times"]:
                inf_msg += (
                    f", convert={np.mean(self.stats['convert_times']) * 1000:.1f}ms"
                )

        pub_msg = "Publish n/a"
        if self.stats["publish_intervals"]:
            pub_hz = 1.0 / np.mean(self.stats["publish_intervals"])
            pub_msg = f"Publish {pub_hz:.1f}Hz (target {self.publish_hz:.1f}Hz)"

        bb_ms = np.mean(self.stats["backbone_times_ms"]) if self.stats["backbone_times_ms"] else 0
        dec_ms = np.mean(self.stats["decoders_times_ms"]) if self.stats["decoders_times_ms"] else 0
        det_ms = np.mean(self.stats["detection_times_ms"]) if self.stats["detection_times_ms"] else 0
        breakdown = f", bb={bb_ms:.1f}ms dec={dec_ms:.1f}ms det={det_ms:.1f}ms"
        logger.info(
            f"Final stats: {inf_msg}{breakdown}, {pub_msg}, "
            f"published={self.stats['publish_count']}, "
            f"interp={self.stats['publish_interpolated_count']}, "
            f"fallback={self.stats['publish_fallback_count']}, "
            f"capture_drop={self.stats['dropped_capture_count']}"
        )
        self._save_stats_csv()
        self._final_stats_logged = True

    def _save_stats_csv(self):
        import csv, datetime

        def finite_array(values):
            arr = np.asarray(values, dtype=np.float64)
            return arr[np.isfinite(arr)]

        def mean_or_nan(values):
            arr = finite_array(values)
            return float(np.mean(arr)) if arr.size else float("nan")

        def median_or_nan(values):
            arr = finite_array(values)
            return float(np.median(arr)) if arr.size else float("nan")

        def variance_or_nan(values):
            arr = finite_array(values)
            return float(np.var(arr)) if arr.size else float("nan")

        def fps_values(ms_values):
            arr = finite_array(ms_values)
            arr = arr[arr > 0]
            return 1000.0 / arr if arr.size else arr

        def prepare_csv_path(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            return path

        frame_header = [
            "infer_index",
            "source_timestamp",
            "persons",
            "infer_ms",
            "backbone_ms",
            "decoders_ms",
            "detection_ms",
            "fov_ms",
            "convert_ms",
            "send_ms",
            "total_ms",
            "ue_sent",
            "status",
        ]
        frame_stats = list(self.stats.get("frame_stats", []))
        frame_csv = prepare_csv_path(
            self.metrics_csv or "output/benchmark_frame_stats.csv"
        )
        with open(frame_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=frame_header)
            writer.writeheader()
            for row in frame_stats:
                writer.writerow({key: row.get(key, "") for key in frame_header})
        logger.info(f"Frame stats saved to {frame_csv}")

        sent_rows = [row for row in frame_stats if row.get("ue_sent") == 1]
        infer_ms_values = [row.get("infer_ms", np.nan) for row in frame_stats]
        total_ms_values = [row.get("total_ms", np.nan) for row in sent_rows]
        backbone_ms_values = [row.get("backbone_ms", np.nan) for row in frame_stats]
        decoders_ms_values = [row.get("decoders_ms", np.nan) for row in frame_stats]
        detection_ms_values = [row.get("detection_ms", np.nan) for row in frame_stats]
        fov_ms_values = [row.get("fov_ms", np.nan) for row in frame_stats]
        convert_ms_values = [row.get("convert_ms", np.nan) for row in sent_rows]
        send_ms_values = [row.get("send_ms", np.nan) for row in sent_rows]

        os.makedirs("output", exist_ok=True)
        path = prepare_csv_path(self.summary_csv or "output/benchmark_stats.csv")
        backbone_path = os.environ.get("TRT_BACKBONE_PATH", "")
        fov_path = os.environ.get("FOV_TRT_PATH", "")
        pub_hz = 1.0 / np.mean(self.stats["publish_intervals"]) if self.stats["publish_intervals"] else float("nan")
        ue_fps = (
            self.stats.get("ue_send_count", 0)
            / max(sum(self.stats["inference_times"]), 1e-9)
            if self.stats["inference_times"]
            else float("nan")
        )
        summary_header = [
            "timestamp",
            "backbone_engine",
            "fov_engine",
            "infer_ms_mean",
            "infer_ms_median",
            "infer_ms_variance",
            "infer_fps_mean",
            "infer_fps_median",
            "infer_fps_variance",
            "total_ms_mean",
            "total_ms_median",
            "total_ms_variance",
            "total_fps_mean",
            "total_fps_median",
            "total_fps_variance",
            "backbone_ms_mean",
            "backbone_ms_median",
            "decoders_ms_mean",
            "decoders_ms_median",
            "detection_ms_mean",
            "detection_ms_median",
            "fov_ms_mean",
            "fov_ms_median",
            "convert_ms_mean",
            "convert_ms_median",
            "send_ms_mean",
            "send_ms_median",
            "publish_hz",
            "ue_send_fps",
            "published",
            "fallback",
            "capture_drop",
            "infer_count",
            "ue_send_count",
            "no_send_count",
            "frame_stats_csv",
        ]
        if os.path.exists(path):
            with open(path, newline="") as f:
                first_row = next(csv.reader(f), [])
            if first_row and first_row != summary_header:
                stem, ext = os.path.splitext(path)
                path = f"{stem}_with_median{ext or '.csv'}"
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(summary_header)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                os.path.basename(backbone_path),
                os.path.basename(fov_path),
                f"{mean_or_nan(infer_ms_values):.2f}",
                f"{median_or_nan(infer_ms_values):.2f}",
                f"{variance_or_nan(infer_ms_values):.2f}",
                f"{mean_or_nan(fps_values(infer_ms_values)):.2f}",
                f"{median_or_nan(fps_values(infer_ms_values)):.2f}",
                f"{variance_or_nan(fps_values(infer_ms_values)):.2f}",
                f"{mean_or_nan(total_ms_values):.2f}",
                f"{median_or_nan(total_ms_values):.2f}",
                f"{variance_or_nan(total_ms_values):.2f}",
                f"{mean_or_nan(fps_values(total_ms_values)):.2f}",
                f"{median_or_nan(fps_values(total_ms_values)):.2f}",
                f"{variance_or_nan(fps_values(total_ms_values)):.2f}",
                f"{mean_or_nan(backbone_ms_values):.2f}",
                f"{median_or_nan(backbone_ms_values):.2f}",
                f"{mean_or_nan(decoders_ms_values):.2f}",
                f"{median_or_nan(decoders_ms_values):.2f}",
                f"{mean_or_nan(detection_ms_values):.2f}",
                f"{median_or_nan(detection_ms_values):.2f}",
                f"{mean_or_nan(fov_ms_values):.2f}",
                f"{median_or_nan(fov_ms_values):.2f}",
                f"{mean_or_nan(convert_ms_values):.2f}",
                f"{median_or_nan(convert_ms_values):.2f}",
                f"{mean_or_nan(send_ms_values):.2f}",
                f"{median_or_nan(send_ms_values):.2f}",
                f"{pub_hz:.2f}",
                f"{ue_fps:.2f}",
                self.stats["publish_count"],
                self.stats["publish_fallback_count"],
                self.stats["dropped_capture_count"],
                self.stats["infer_count"],
                self.stats["ue_send_count"],
                self.stats["infer_count"] - self.stats["ue_send_count"],
                frame_csv,
            ])
        logger.info(f"Stats saved to {path}")

    def _release_video_source_with_timeout(self, timeout_s=10.0):
        release_errors = []

        def _release():
            try:
                self.video_source.release()
            except Exception as exc:
                release_errors.append(exc)

        t = threading.Thread(target=_release, daemon=True)
        t.start()
        t.join(timeout=timeout_s)

        if t.is_alive():
            logger.warning("Timed out while releasing video source; continue shutdown")
        elif release_errors:
            logger.warning(f"Video source release raised: {release_errors[0]}")

    def start(self):
        logger.info("Starting realtime publisher (Press Ctrl+C to stop)")
        self.running = True

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)

        if self.record:
            self.recording_thread = threading.Thread(
                target=self._recording_loop, daemon=True
            )
            self.recording_thread.start()

        self.capture_thread.start()
        self.inference_thread.start()
        self.publish_thread.start()

        while self.running:
            if self.video_ended and not self.inference_thread.is_alive():
                self.running = False
                break
            if (
                not self.capture_thread.is_alive()
                and not self.inference_thread.is_alive()
            ):
                self.running = False
                break
            time.sleep(0.05)

        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)

        self._log_final_stats()

    def stop(self):
        if self._closed:
            return

        self.running = False
        self._release_video_source_with_timeout(timeout_s=1.0)

        if (
            self.record
            and self.recording_thread is not None
            and self.recording_thread.is_alive()
        ):
            logger.info("Waiting for recording thread to finish writing to disk...")
            self.recording_thread.join(timeout=5.0)

        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        if self.publish_thread is not None and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)

        self._log_final_stats()

        self.publisher.close()
        if self.ue_sender is not None:
            self.ue_sender.close()
        self._closed = True


def main():
    parser = argparse.ArgumentParser(
        description="Publish SAM 3D pose over ZMQ + Unreal Engine UDP/TCP"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="camera",
        choices=["camera", "video"],
        help="Video source type",
    )
    parser.add_argument(
        "--video", type=str, help="Path to video file (for --source video)"
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        help="Camera intrinsics JSON path (required for --source video)",
    )
    parser.add_argument(
        "--no-loop",
        dest="no_loop",
        action="store_true",
        help="Disable loop video playback (for --source video)",
    )
    parser.add_argument(
        "--publish-hz", type=float, default=50.0, help="Publisher output rate in Hz"
    )
    parser.add_argument(
        "--interp-lag-ms",
        type=float,
        default=140.0,
        help="Interpolation lag in ms",
    )
    parser.add_argument(
        "--addr", type=str, default="tcp://*:5556", help="ZMQ bind address"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        choices=[256, 384, 512],
        help="Image size for SAM3D model",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=YOLO_MODEL_PATH,
        help="YOLO pose model path",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=str,
        required=True,
        help="SMPL model pickle file",
    )
    parser.add_argument(
        "--nn-model-dir",
        type=str,
        required=True,
        help="NN fusion model directory",
    )
    parser.add_argument(
        "--mhr2smpl-mapping-path",
        type=str,
        required=True,
        help="Path to mhr2smpl_mapping.npz",
    )
    parser.add_argument(
        "--mhr-mesh-path",
        type=str,
        default=None,
        help="Path to MHR mesh PLY",
    )
    parser.add_argument(
        "--smoother-dir",
        type=str,
        default=None,
        help="Smoother checkpoint directory",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable recording raw video and SMPL data to disk",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="output/records",
        help="Directory to save recordings",
    )
    parser.add_argument(
        "--min-person-confidence",
        type=float,
        default=0.75,
        help="Minimum YOLO detection confidence",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Per-inference-frame timing CSV path",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Summary timing CSV path with mean/median/variance",
    )

    # --- UE 通信オプション ---
    ue_group = parser.add_argument_group("Unreal Engine streaming")
    ue_group.add_argument(
        "--ue",
        type=str,
        choices=["udp", "tcp"],
        default=None,
        help=(
            "Enable UE streaming. "
            "'udp': wait for HELLO then send datagrams (SAT-HMR compatible). "
            "'tcp': listen and accept UE connection."
        ),
    )
    ue_group.add_argument(
        "--ue-port",
        type=int,
        default=62004,
        help="[UDP only] UE receive port (default: 62004)",
    )
    ue_group.add_argument(
        "--local-port",
        type=int,
        default=5555,
        help="Local port: UDP=HELLO listen port, TCP=listen port (default: 5555)",
    )

    args = parser.parse_args()

    if args.publish_hz <= 0:
        parser.error("--publish-hz must be > 0")
    if args.interp_lag_ms < 0:
        parser.error("--interp-lag-ms must be >= 0")

    # --- UE sender 初期化 ---
    ue_sender = None
    if args.ue is not None:
        ue_sender = UESender(
            protocol=args.ue,
            ue_port=args.ue_port,
            local_port=args.local_port,
        )
        logger.info(
            f"UE sender ({args.ue.upper()}) created. "
            f"Waiting for connection in background..."
        )
        ue_sender.wait_for_connection()

    # --- Video source ---
    if args.source == "camera":
        logger.info("Video source: camera (realtime mode; runs until Ctrl+C)")
        video_source = create_video_source("camera", width=848, height=480, fps=30)
    else:
        if not args.video:
            parser.error("--video required when --source video")
        if not args.no_loop:
            logger.info("Video source: file (loop playback enabled; add --no-loop to exit at EOF)")
        else:
            logger.info("Video source: file (single pass; exits at EOF)")
        video_source = create_video_source(
            "video",
            video_path=args.video,
            intrinsics_path=args.intrinsics,
            loop=not args.no_loop,
        )

    publisher = RealtimePublisherUE(
        video_source=video_source,
        publish_hz=args.publish_hz,
        interpolate_lag_ms=args.interp_lag_ms,
        smpl_model_path=args.smpl_model_path,
        multiview_model_dir=args.nn_model_dir,
        mhr2smpl_mapping_path=args.mhr2smpl_mapping_path,
        mhr_mesh_path=args.mhr_mesh_path,
        smoother_dir=args.smoother_dir,
        addr=args.addr,
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
        record=args.record,
        record_dir=args.record_dir,
        min_person_confidence=args.min_person_confidence,
        ue_sender=ue_sender,
        metrics_csv=args.metrics_csv,
        summary_csv=args.summary_csv,
    )

    try:
        publisher.start()
    except KeyboardInterrupt:
        logger.info("Stopping...")
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        publisher.stop()
        logger.success("Stopped.")


if __name__ == "__main__":
    main()
