"""Source-video identity and camera-observation checks shared by the ue/ pipeline scripts (stdlib only)."""
from fractions import Fraction
import json
import math
from pathlib import Path


def check_source_stat(path,size=None,mtime_ns=None):
    """Return the source's stat after checking the probed size / mtime (``None`` skips that check)."""
    stat=Path(path).stat()
    if size is not None and stat.st_size!=size:
        raise RuntimeError(f'Source size changed: expected {size}, found {stat.st_size}')
    if mtime_ns is not None and stat.st_mtime_ns!=mtime_ns:
        raise RuntimeError(f'Source mtime changed: expected {mtime_ns}, found {stat.st_mtime_ns}')
    return stat


def capture_properties(video):
    """Frame count, size and FPS (Fraction) of an opened cv2.VideoCapture."""
    import cv2
    return {'frame_count':int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width':int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height':int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps':Fraction(str(video.get(cv2.CAP_PROP_FPS))).limit_denominator(1000000)}


def check_capture(decoded,expected):
    """Raise if a decoded property differs from its expected value (``None`` skips that check)."""
    for key,value in expected.items():
        if value is None:continue
        same=decoded[key]==Fraction(value) if key=='fps' else decoded[key]==value
        if not same:raise RuntimeError(f'Decoded source {key} changed: expected {value}, found {decoded[key]}')


def finite_numbers(values):
    return all(isinstance(item,(int,float)) and math.isfinite(item) for item in values)


def load_video_observations(path,expected):
    """Load observe_video_camera.py output, checking its source identity against ``expected``."""
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('type')!='video_camera_observations' or value.get('version')!=1:
        raise RuntimeError('Unsupported video camera observations')
    identity=value.get('source',{})
    for key,expected_value in expected.items():
        same=(Fraction(identity.get('fps','0'))==Fraction(expected_value) if key=='fps'
              else identity.get(key)==expected_value)
        if not same:raise RuntimeError('Video camera observation source mismatch: '+key)
    intrinsics=value.get('intrinsics')
    if not isinstance(intrinsics,list) or len(intrinsics)!=4 or not finite_numbers(intrinsics) or intrinsics[0]<=0 or intrinsics[1]<=0:
        raise RuntimeError('Video camera observations contain invalid intrinsics')
    frames=value.get('frames')
    if not isinstance(frames,list) or len(frames)!=expected['frame_count']:
        raise RuntimeError('Video camera observations do not cover the source')
    for index,frame in enumerate(frames):
        if not isinstance(frame,dict):raise RuntimeError('Video camera observation frame must be an object')
        body=frame.get('body12')
        if frame.get('frame_index')!=index or not isinstance(body,list) or len(body)!=12:
            raise RuntimeError('Video camera observation frames are not contiguous body12 records')
        for point in body:
            if not isinstance(point,list) or len(point)!=3 or not finite_numbers(point) or not 0<=point[2]<=1:
                raise RuntimeError('Video camera observations contain invalid body keypoints')
    return value
