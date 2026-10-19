"""Run the complete repository pipeline with one packaged Unreal scene."""
import argparse
import json
from pathlib import Path

from run_pipeline import main as run_pipeline


SCENE_MODES={'normal','rockdrop','orbit'}


def absolute_path(path):
    return str(Path(path).expanduser().resolve())


def package_settings(package):
    root=Path(package).expanduser().resolve()
    manifest_path=root/'package-manifest.json'
    game=root/'Windows'/'MyProject_ROMP.exe'
    if not root.is_dir():raise FileNotFoundError(f'Package directory was not found: {root}')
    if not manifest_path.is_file():raise FileNotFoundError(f'Package manifest was not found: {manifest_path}')
    if not game.is_file():raise FileNotFoundError(f'Packaged game was not found: {game}')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    scene_mode=manifest.get('scene_mode')
    if scene_mode not in SCENE_MODES:raise ValueError(f'Invalid scene_mode in package manifest: {scene_mode}')
    if manifest.get('input')!='external_video':raise ValueError('Package does not declare external_video input')
    return root,game,manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,help='Normal, RockDrop, or SphereOrbit package directory')
    p.add_argument('--video',required=True,help='Input video')
    p.add_argument('--output',help='New output directory (default: results/ue/<mode>/Recordings/<timestamp>)')
    p.add_argument('--observation-python',help='Optional separate Python for camera observation')
    p.add_argument('--ffmpeg',default='ffmpeg',help='Native ffmpeg for probing and compositing')
    p.add_argument('--ffprobe',default='ffprobe',help='Native ffprobe for probing and validation')
    p.add_argument('--recorder-ffmpeg',help='Windows ffmpeg used by the Win64 Unreal package')
    p.add_argument('--camera-calibration')
    p.add_argument('--timeout',type=float,default=7200)
    p.add_argument('--frame-timeout',type=float,default=600)
    p.add_argument('--max-missing-frames',type=int,default=0)
    pose=p.add_mutually_exclusive_group()
    pose.add_argument('--pose-cache')
    pose.add_argument('--replay-poses')
    a=p.parse_args(argv)

    _,game,manifest=package_settings(a.package)
    video=Path(a.video).expanduser().resolve()
    if not video.is_file():p.error(f'Input video was not found: {video}')

    pipeline_args=['--packaged-exe',str(game),'--source',str(video),
                   '--scene-mode',manifest['scene_mode'],'--repeat','1','--despill',
                   '--ffmpeg',a.ffmpeg,'--ffprobe',a.ffprobe,
                   '--timeout',str(a.timeout),'--frame-timeout',str(a.frame_timeout),
                   '--max-missing-frames',str(a.max_missing_frames)]
    if manifest.get('auto_camera') is True:pipeline_args.append('--auto-camera')
    if a.output:pipeline_args.extend(('--output-dir',absolute_path(a.output)))
    if a.observation_python:pipeline_args.extend(('--observation-python',a.observation_python))
    if a.recorder_ffmpeg:pipeline_args.extend(('--recorder-ffmpeg',a.recorder_ffmpeg))
    if a.camera_calibration:pipeline_args.extend(('--camera-calibration',absolute_path(a.camera_calibration)))
    if a.pose_cache:pipeline_args.extend(('--pose-cache',absolute_path(a.pose_cache)))
    if a.replay_poses:pipeline_args.extend(('--replay-poses',absolute_path(a.replay_poses)))
    run_pipeline(pipeline_args)


if __name__=='__main__':main()
