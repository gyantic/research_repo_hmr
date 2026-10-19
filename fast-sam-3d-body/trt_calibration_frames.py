"""Calibration-frame sampling shared by the INT8 TensorRT converters.

Used by convert_backbone_tensorrt.py (modelopt PTQ) and
convert_moge_encoder_trt.py (IInt8EntropyCalibrator2). Each converter applies
its own dtype / normalization to the returned frames.
"""

import os

DEFAULT_CALIB_VIDEO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "videos", "sample_video.mp4"
)
DEFAULT_NUM_CALIB_FRAMES = 200


def sample_video_frames_rgb(video_path, num_frames, width, height):
    """Return up to ``num_frames`` evenly spaced RGB uint8 frames resized to (height, width, 3)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)
    frames = []
    for frame_idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (width, height)))
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames
