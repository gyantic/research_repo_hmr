"""Estimate source-video intrinsics independently of SAT-HMR."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from source_checks import capture_properties,check_capture,check_source_stat

DEFAULT_MOGE=str(Path(__file__).resolve().parent.parent/'checkpoints'/'moge2'/'model.pt')

def finite_intrinsics(values,width,height):
    values=[float(value) for value in values]
    if len(values)!=4 or not all(math.isfinite(value) for value in values):
        raise ValueError('Intrinsics must contain four finite values')
    fx,fy,cx,cy=values
    if fx<=0 or fy<=0 or not 0<=cx<width or not 0<=cy<height:
        raise ValueError('Intrinsics are outside the source image')
    return values

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--source-identity',required=True);parser.add_argument('--source-size',required=True,type=int)
    parser.add_argument('--source-mtime-ns',required=True,type=int);parser.add_argument('--frame-count',required=True,type=int)
    parser.add_argument('--width',required=True,type=int);parser.add_argument('--height',required=True,type=int)
    parser.add_argument('--fps',required=True)
    parser.add_argument('--moge-model',default=DEFAULT_MOGE)
    args=parser.parse_args()
    source=Path(args.source);stat=check_source_stat(source,args.source_size)
    video=cv2.VideoCapture(str(source))
    if not video.isOpened():raise RuntimeError('Cannot open source video for camera observation')
    decoded=capture_properties(video)
    check_capture(decoded,{'frame_count':args.frame_count,'width':args.width,'height':args.height,'fps':args.fps})
    count,width,height,fps=decoded['frame_count'],decoded['width'],decoded['height'],decoded['fps']

    representative=sorted(set(int(round(value)) for value in np.linspace(0,max(0,count-1),7)))
    import torch
    from moge.model.v2 import MoGeModel
    moge=MoGeModel.from_pretrained(args.moge_model).cuda().eval()
    frames=[];intrinsic_samples=[]
    try:
        for frame_index in range(count):
            ok,frame=video.read()
            if not ok:raise RuntimeError(f'Could not decode source frame {frame_index}')
            # Zero-confidence placeholders: no 2D joints are detected here, but UE accepts body_keypoints_2d only as 12 [x,y,confidence] points.
            frames.append({'frame_index':frame_index,'body12':[[0.0,0.0,0.0] for _ in range(12)]})
            if frame_index in representative:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image=torch.from_numpy(rgb).permute(2,0,1).cuda().float()/255.0
                with torch.inference_mode():prediction=moge.infer(image,resolution_level=0,force_projection=True,apply_mask=True,use_fp16=True)
                matrix=prediction['intrinsics'].detach().double().cpu().numpy()
                intrinsic_samples.append(finite_intrinsics((matrix[0,0]*width,matrix[1,1]*height,matrix[0,2]*width,matrix[1,2]*height),width,height))
    finally:
        video.release()
    if len(frames)!=count:raise RuntimeError('Camera observations do not cover every source frame')
    check_source_stat(source,stat.st_size,stat.st_mtime_ns)  # unchanged while observing
    if not intrinsic_samples:raise RuntimeError('MoGe produced no intrinsic samples')
    intrinsics=finite_intrinsics(np.median(np.asarray(intrinsic_samples),axis=0),width,height)
    output={
        'type':'video_camera_observations','version':1,
        'source':{'path':args.source_identity,'size':args.source_size,'mtime_ns':args.source_mtime_ns,
                  'frame_count':count,'width':width,'height':height,'fps':str(fps)},
        'intrinsics':intrinsics,'intrinsics_source':'moge2_median',
        'representative_frame_indices':representative,
        'intrinsic_samples':intrinsic_samples,'body_joint_order':[
            'L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist',
            'L_Hip','R_Hip','L_Knee','R_Knee','L_Ankle','R_Ankle'],
        'frames':frames}
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:json.dump(output,stream,separators=(',',':'),allow_nan=False)

if __name__=='__main__':main()
