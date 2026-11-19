"""Run the complete repository pipeline with one packaged Unreal scene."""
import argparse
import json
from pathlib import Path

from run_pipeline import SCENE_MODES,absolute_path,add_common_args,forward_common_args
from run_pipeline import main as run_pipeline


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
    common=add_common_args(p)
    a=p.parse_args(argv)

    _,game,manifest=package_settings(a.package)
    video=Path(a.video).expanduser().resolve()
    if not video.is_file():p.error(f'Input video was not found: {video}')

    pipeline_args=['--packaged-exe',str(game),'--source',str(video),
                   '--scene-mode',manifest['scene_mode'],'--despill',*forward_common_args(a,common)]
    if manifest.get('auto_camera') is True:pipeline_args.append('--auto-camera')
    if a.output:pipeline_args.extend(('--output-dir',absolute_path(a.output)))
    run_pipeline(pipeline_args)


if __name__=='__main__':main()
