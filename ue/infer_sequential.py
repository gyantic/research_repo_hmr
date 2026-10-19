"""Run in the SAT-HMR WSL environment; one source frame -> one UE ACK."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys

CACHE_VERSION=1
SMPL_PARENTS=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)

def model_bone_lengths_m(joints):
    if len(joints)<len(SMPL_PARENTS):
        raise ValueError('SAT-HMR must provide at least 24 SMPL joints')
    if len(joints[0])!=3 or not all(math.isfinite(float(value)) for value in joints[0]) or float(joints[0][2])<=0:
        raise ValueError('SMPL pelvis must be finite with positive forward depth')
    result=[]
    for child in range(1,len(SMPL_PARENTS)):
        parent=SMPL_PARENTS[child]
        if len(joints[child])!=3 or len(joints[parent])!=3:
            raise ValueError('Each SMPL joint must contain three coordinates')
        delta=[float(joints[child][axis])-float(joints[parent][axis]) for axis in range(3)]
        if not all(math.isfinite(value) for value in delta):
            raise ValueError('SMPL joint coordinates must be finite')
        length=math.sqrt(sum(value*value for value in delta))
        if not math.isfinite(length) or length<=0:
            raise ValueError('SMPL bone lengths must be finite and positive')
        result.append(length)
    return result

def json_line(value):
    return json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'

def source_fingerprint(args,frame_count,fps):
    stat=Path(args.source).stat()
    if args.source_size is not None and stat.st_size!=args.source_size:
        raise RuntimeError(f'Source size changed: expected {args.source_size}, found {stat.st_size}')
    if args.source_mtime_ns is not None and stat.st_mtime_ns!=args.source_mtime_ns:
        raise RuntimeError(f'Source mtime changed: expected {args.source_mtime_ns}, found {stat.st_mtime_ns}')
    return {'path':args.source_identity or str(Path(args.source).resolve()),'size':stat.st_size,
            'mtime_ns':stat.st_mtime_ns,'frame_count':frame_count,'fps':str(fps)}

class PoseCacheWriter:
    def __init__(self,path,source):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.partial=self.path.with_name(self.path.name+f'.{os.getpid()}.partial')
        self.stream=self.partial.open('x',encoding='utf-8',newline='\n')
        self.digest=hashlib.sha256();self.count=0
        self._write({'type':'metadata','version':CACHE_VERSION,'source':source})
    def _write(self,value):
        line=json_line(value);self.stream.write(line);self.stream.flush();self.digest.update(line.encode())
    def write_pose(self,frame_index,packet,held):
        if frame_index!=self.count:raise RuntimeError('Pose cache frame indices must be contiguous')
        self._write({'type':'pose','frame_index':frame_index,'held':held,**packet});self.count+=1
    def finish(self,held_indices):
        footer={'type':'complete','frame_count':self.count,'held_frame_indices':held_indices,'sha256':self.digest.hexdigest()}
        self.stream.write(json_line(footer));self.stream.flush();os.fsync(self.stream.fileno());self.stream.close()
        os.replace(self.partial,self.path)
    def close(self):
        if not self.stream.closed:self.stream.close()

def load_pose_cache(path):
    cache=Path(path)
    if '.partial' in cache.suffixes:raise RuntimeError('Refusing a partial pose cache')
    digest=hashlib.sha256();poses=[];metadata=None;footer=None
    with cache.open('r',encoding='utf-8') as stream:
        for line_number,line in enumerate(stream,1):
            if not line.strip():raise RuntimeError(f'Blank line in pose cache at line {line_number}')
            value=json.loads(line);kind=value.get('type')
            if footer is not None:raise RuntimeError('Pose cache has data after its complete footer')
            if kind=='metadata' and metadata is None and not poses:
                metadata=value;digest.update(line.encode())
            elif kind=='pose' and metadata is not None:
                if value.get('frame_index')!=len(poses):raise RuntimeError('Pose cache frame indices are not contiguous')
                poses.append(value);digest.update(line.encode())
            elif kind=='complete' and metadata is not None:footer=value
            else:raise RuntimeError(f'Invalid pose cache record at line {line_number}')
    if metadata is None or footer is None:raise RuntimeError('Pose cache is incomplete')
    if metadata.get('version')!=CACHE_VERSION:raise RuntimeError('Unsupported pose cache version')
    if footer.get('frame_count')!=len(poses):raise RuntimeError('Pose cache complete footer has the wrong frame count')
    if footer.get('sha256')!=digest.hexdigest():raise RuntimeError('Pose cache checksum does not match')
    held=[pose['frame_index'] for pose in poses if pose.get('held')]
    if footer.get('held_frame_indices')!=held:raise RuntimeError('Pose cache held-frame manifest does not match')
    return metadata['source'],poses

def load_observations(path,current_source):
    if not path:return None
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('type')!='video_camera_observations' or value.get('version')!=1:
        raise RuntimeError('Unsupported video camera observations')
    source=value.get('source',{})
    for key in ('path','size','mtime_ns','frame_count','fps'):
        if source.get(key)!=current_source.get(key):raise RuntimeError('Video camera observation source mismatch: '+key)
    intrinsics=value.get('intrinsics');frames=value.get('frames')
    if not isinstance(intrinsics,list) or len(intrinsics)!=4 or not isinstance(frames,list) or len(frames)!=current_source['frame_count']:
        raise RuntimeError('Invalid video camera observation dimensions')
    return value

def replay_model_intrinsics(pose,observations):
    value=pose.get('model_intrinsics')
    return pose.get('intrinsics') if observations is not None and value is None else value

def replay(args):
    from pose_rendezvous import PoseRendezvous
    source,poses=load_pose_cache(args.replay_poses)
    if args.source_frame_count is None or args.source_fps is None:
        raise RuntimeError('--replay-poses requires --source-frame-count and --source-fps')
    current=source_fingerprint(args,args.source_frame_count,Fraction(args.source_fps))
    if current!=source:
        fields=','.join(key for key in current if current.get(key)!=source.get(key))
        raise RuntimeError('Pose cache source identity mismatch: '+fields)
    observations=load_observations(args.observations,current)
    bridge=PoseRendezvous(args.inbox,args.timeout)
    for pose in poses:
        frame_index=pose['frame_index']
        if observations:
            observed=observations['frames'][frame_index];intrinsics=observations['intrinsics']
            body_keypoints_2d=observed.get('body12') if observed else pose.get('body_keypoints_2d')
        else:
            intrinsics=pose['intrinsics'];body_keypoints_2d=pose.get('body_keypoints_2d')
        bridge.submit(pose['quaternions'],pose['pelvis_camera_m'],intrinsics,
                      pose.get('model_bone_lengths_m'),pose.get('model_joints_camera_m'),
                      replay_model_intrinsics(pose,observations),body_keypoints_2d)
        print(json.dumps({'acknowledged_frame':frame_index,'source_frames':len(poses),'replayed':True}),flush=True)
    bridge.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sat-root',default=str(Path(__file__).resolve().parent.parent/'sat-hmr'))
    parser.add_argument('--source',required=True)
    parser.add_argument('--source-identity');parser.add_argument('--source-size',type=int);parser.add_argument('--source-mtime-ns',type=int)
    parser.add_argument('--source-frame-count',type=int);parser.add_argument('--source-fps')
    parser.add_argument('--observations',help='Source-locked independent intrinsics and body12 observations')
    parser.add_argument('--inbox',required=True)
    parser.add_argument('--image-size',type=int,default=1288)
    parser.add_argument('--max-frames',type=int,default=0,help='Smoke only: composite against an equally trimmed source video')
    parser.add_argument('--confidence',type=float,default=.3)
    parser.add_argument('--timeout',type=float,default=600)
    cache_group=parser.add_mutually_exclusive_group()
    cache_group.add_argument('--pose-cache',help='Atomically write a complete per-frame pose JSONL cache')
    cache_group.add_argument('--replay-poses',help='Replay a complete pose JSONL cache without SAT-HMR/Torch')
    parser.add_argument('--max-missing-frames',type=int,default=0,help='Hold the previous pose across this many consecutive missing detections')
    args=parser.parse_args()
    if args.max_missing_frames<0:parser.error('--max-missing-frames must be non-negative')
    if args.pose_cache and args.max_frames:parser.error('--pose-cache cannot be combined with --max-frames')
    if args.replay_poses and (args.max_frames or args.max_missing_frames):
        parser.error('--max-frames/--max-missing-frames apply only to inference')
    if args.replay_poses:return replay(args)
    sys.path.insert(0,args.sat_root);os.chdir(args.sat_root)
    import cv2
    import torch
    import video_demo as vd
    from pose_rendezvous import PoseRendezvous
    config=vd.load_config(args.image_size)
    model,_=vd.build_sat_model(config,set_criterion=False)
    state=torch.load('weights/sat_hmr/sat_644.pth',map_location='cpu',weights_only=False)
    state,_=vd.dequantize_int8_weight_state_dict_if_needed(state)
    model.load_state_dict(state,strict=False);model.cuda().eval()
    bridge=PoseRendezvous(args.inbox,args.timeout)
    video=cv2.VideoCapture(args.source)
    if not video.isOpened():raise RuntimeError('Cannot open source video')
    count=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=Fraction(str(video.get(cv2.CAP_PROP_FPS))).limit_denominator(1000000)
    if args.source_frame_count is not None and count!=args.source_frame_count:
        raise RuntimeError(f'Source frame count changed: expected {args.source_frame_count}, found {count}')
    if args.source_fps is not None and fps!=Fraction(args.source_fps):
        raise RuntimeError(f'Source FPS changed: expected {args.source_fps}, found {fps}')
    fingerprint=source_fingerprint(args,count,fps)
    observations=load_observations(args.observations,fingerprint)
    cache=PoseCacheWriter(args.pose_cache,fingerprint) if args.pose_cache else None
    previous_translation=None;last_packet=None;processed=0;missing_run=0;held_indices=[]
    try:
        while not args.max_frames or processed<args.max_frames:
            ok,frame=video.read()
            if not ok:break
            image,model_input_size,original_size=vd.preprocess_frame(frame,args.image_size)
            with torch.inference_mode():
                prediction=model([image.cuda()],[{'img_size':torch.tensor(model_input_size,device='cuda'),'img_path':'sample'}])
            confidence=prediction['pred_confs'][0].reshape(-1)
            candidates=torch.nonzero(confidence>=args.confidence).reshape(-1)
            held=False
            if candidates.numel()==0:
                missing_run+=1
                if last_packet is None:raise RuntimeError(f'No reliable SMPL detection at frame {processed}; no previous pose can be held')
                if missing_run>args.max_missing_frames:
                    raise RuntimeError(f'No reliable SMPL detection at frame {processed}; missing run {missing_run} exceeds --max-missing-frames {args.max_missing_frames}')
                packet=dict(last_packet);held=True;held_indices.append(processed)
                print(json.dumps({'held_frame_index':processed,'missing_run':missing_run}),flush=True)
            else:
                missing_run=0;translations=prediction['pred_transl'][0]
                # Follow the person nearest the previous translation so multi-person clips keep one identity.
                if previous_translation is None:index=int(candidates[torch.argmax(confidence[candidates])])
                else:index=int(candidates[torch.argmin(torch.linalg.vector_norm(translations[candidates]-previous_translation,dim=-1))])
                previous_translation=translations[index].detach()
                pose=prediction['pred_poses'][0][index].cpu().numpy()
                intrinsic_matrix=prediction['pred_intrinsics'][0].reshape(3,3).cpu().numpy().copy()
                # K is predicted for the resized model input; rescale its x/y rows to source-video pixels.
                intrinsic_matrix[0]*=original_size[1]/model_input_size[1];intrinsic_matrix[1]*=original_size[0]/model_input_size[0]
                model_intrinsics=[float(intrinsic_matrix[0,0]),float(intrinsic_matrix[1,1]),float(intrinsic_matrix[0,2]),float(intrinsic_matrix[1,2])]
                model_joints=prediction['pred_j3ds'][0][index][:24].cpu().tolist()
                packet={'quaternions':vd.smpl_pose_to_ue_quaternions(pose).tolist(),
                        'pelvis_camera_m':model_joints[0],
                        'model_bone_lengths_m':model_bone_lengths_m(model_joints),
                        'model_joints_camera_m':model_joints,
                        'model_intrinsics':model_intrinsics,
                        'intrinsics':model_intrinsics}
                last_packet=packet
            if observations:
                observed=observations['frames'][processed]
                if observed.get('frame_index')!=processed:raise RuntimeError('Video camera observations are not contiguous')
                packet['intrinsics']=observations['intrinsics']
                packet['body_keypoints_2d']=observed['body12']
            bridge.submit(packet['quaternions'],packet['pelvis_camera_m'],packet['intrinsics'],
                          packet.get('model_bone_lengths_m'),packet.get('model_joints_camera_m'),
                          packet.get('model_intrinsics'),packet.get('body_keypoints_2d'))
            if cache:cache.write_pose(processed,packet,held)
            processed+=1
            print(json.dumps({'acknowledged_frame':processed-1,'source_frames':count}),flush=True)
        if not args.max_frames and processed!=count:raise RuntimeError(f'Decoded {processed} of {count} source frames')
        bridge.finish()
        if cache:
            if processed!=count:raise RuntimeError('Refusing to finalize a pose cache that does not cover the full source')
            if source_fingerprint(args,count,fps)!=fingerprint:raise RuntimeError('Source changed while pose cache was being produced')
            cache.finish(held_indices)
    finally:
        video.release()
        if cache:cache.close()
if __name__=='__main__':main()
