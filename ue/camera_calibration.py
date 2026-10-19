"""Validate canonical-video intrinsics for pose-driven UE auto-camera recording."""
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path
import re

import numpy as np

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def portable_path(path):
    value=str(path).replace('\\','/')
    match=re.match(r'^/mnt/([a-zA-Z])/(.*)$',value)
    if match:value=f'{match.group(1)}:/{match.group(2)}'
    return os.path.normcase(os.path.normpath(value))

def is_sha256_hex(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-fA-F]{64}',value) is not None

def finite_array(value,shape,name):
    result=np.asarray(value,np.float64)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError(f'{name} must be finite with shape {shape}')
    return result

def validate_source_identity(report,source_info):
    expected=report['source']
    observed={'frame_count':source_info['frames'],'width':source_info['width'],'height':source_info['height']}
    for key,value in observed.items():
        if expected.get(key)!=value:raise ValueError('Canonical sidecar and decoded source differ: '+key)
    expected_fps=float(Fraction(str(expected.get('fps'))))
    observed_fps=float(Fraction(str(source_info['fps'])))
    if not math.isclose(expected_fps,observed_fps,rel_tol=0,abs_tol=1e-6):
        raise ValueError('Canonical sidecar and decoded source differ: fps')

def validate_canonical(source,calibration_override=None):
    source=Path(source).resolve();sidecar=Path(str(source)+'.json')
    if not source.is_file():raise FileNotFoundError(source)
    if not sidecar.is_file():raise ValueError('Auto camera requires the calibrated canonical video sidecar: '+str(sidecar))
    metadata=json.loads(sidecar.read_text(encoding='utf-8'))
    if portable_path(metadata.get('output',''))!=portable_path(source):raise ValueError('Canonical sidecar output does not identify the selected source')
    output_sha256=metadata.get('output_sha256')
    if not is_sha256_hex(output_sha256):
        raise ValueError('Canonical sidecar requires a valid output_sha256')
    if sha256_file(source)!=output_sha256.lower():raise ValueError('Canonical video SHA-256 does not match its sidecar')
    match=re.match(r'^(\d{6}_pose\d+)_canonical\.[^.]+$',source.name)
    if not match:raise ValueError('Canonical source name must identify its CMU sequence')
    sequence=match.group(1);camera=metadata.get('camera')
    if not isinstance(camera,str) or not camera:raise ValueError('Canonical sidecar has no camera name')
    calibration=Path(calibration_override).resolve() if calibration_override else source.parent/f'calibration_{sequence}.json'
    if not calibration.is_file():raise FileNotFoundError(calibration)
    calibration_data=json.loads(calibration.read_text(encoding='utf-8'))
    calibration_sha256=sha256_file(calibration)
    recorded_calibration_sha256=metadata.get('calibration_sha256')
    if recorded_calibration_sha256 is not None:
        if not is_sha256_hex(recorded_calibration_sha256):
            raise ValueError('Canonical sidecar calibration_sha256 is invalid')
        if recorded_calibration_sha256.lower()!=calibration_sha256:
            raise ValueError('Calibration SHA-256 does not match the canonical sidecar')
    matches=[item for item in calibration_data.get('cameras',[]) if item.get('name')==camera]
    if len(matches)!=1:raise ValueError(f'Expected exactly one calibration camera {camera!r}')
    item=matches[0]
    source_k=finite_array(item.get('K'),(3,3),'K');distortion=finite_array(item.get('distCoef'),(5,),'distCoef')
    rotation=finite_array(item.get('R'),(3,3),'R');translation=finite_array(item.get('t'),(3,1),'t').reshape(3)
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-6) or not math.isclose(float(np.linalg.det(rotation)),1.0,abs_tol=1e-6):
        raise ValueError('Calibration R must be a proper orthonormal rotation')
    resolution=item.get('resolution')
    if resolution!=[metadata.get('width'),metadata.get('height')]:raise ValueError('Calibration and canonical video resolutions differ')
    if not np.allclose(source_k,finite_array(metadata.get('source_K'),(3,3),'source_K'),rtol=0,atol=1e-6):
        raise ValueError('Canonical sidecar source_K does not match the calibration camera')
    if not np.allclose(distortion,finite_array(metadata.get('distortion'),(5,),'distortion'),rtol=0,atol=1e-9):
        raise ValueError('Canonical sidecar distortion does not match the calibration camera')
    canonical_k=finite_array(metadata.get('canonical_K'),(3,3),'canonical_K')
    if not np.allclose(canonical_k[2],[0,0,1],atol=1e-12) or canonical_k[0,0]<=0 or canonical_k[1,1]<=0:
        raise ValueError('Canonical intrinsics are invalid')
    stat=source.stat()
    identity={'path':str(source),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':output_sha256.lower(),
              'frame_count':metadata.get('frames'),'width':metadata.get('width'),'height':metadata.get('height'),'fps':metadata.get('fps')}
    runtime={'expected_intrinsics':[float(canonical_k[0,0]),float(canonical_k[1,1]),float(canonical_k[0,2]),float(canonical_k[1,2])]}
    report={'type':'auto_camera_input_validation','version':1,'mode':'sat_hmr_pelvis_camera_translation',
            'source':identity,'sequence':sequence,'camera':camera,
            'canonical_intrinsics_validated':True,
            'calibration_path':str(calibration),'calibration_sha256':calibration_sha256,
            'sidecar_path':str(sidecar),'sidecar_sha256':sha256_file(sidecar),
            'extrinsics_applied':False,
            'extrinsics_usage':'Calibration R/t are validated provenance only. Canonical K drives projection; camera position comes from SAT-HMR 3D translation after SAT-to-UE unit conversion and is not fitted to source joints.',
            'source_K':source_k.tolist(),'distortion':distortion.tolist(),
            'canonical_K':canonical_k.tolist(),'R':rotation.tolist(),'t_cm':translation.tolist(),
            **runtime}
    return runtime,report

