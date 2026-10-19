"""One command: sequential SAT-HMR -> UE Camera AVI -> timed chroma-key composite."""
import argparse
from datetime import datetime
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parent
PACKAGE_NAMES={'normal':'Normal','rockdrop':'RockDrop','orbit':'SphereOrbit'}
RESULTS_ROOT=REPO_ROOT/'results'/'ue'
RECORDER_STARTUP_TIMEOUT_S=180

def executable(value):
    result=shutil.which(value)
    if not result:raise RuntimeError(f'Missing executable {value}; install it or provide its full path')
    return result

def windows_path(path):
    """Return a path usable by the Win64 recorder, without relocating repository files."""
    resolved=Path(path).resolve()
    if os.name=='nt':return str(resolved)
    converter=executable('wslpath')
    if not shutil.which('powershell.exe'):
        raise RuntimeError('A Win64 Unreal package requires Windows interop (powershell.exe and wslpath)')
    result=subprocess.run([converter,'-a','-w',str(resolved)],check=True,capture_output=True,text=True)
    value=result.stdout.strip()
    if not value:raise RuntimeError('wslpath returned an empty Windows path for '+str(resolved))
    return value

def find_windows_executable(value):
    """Resolve a Windows executable used by the Win64 recorder."""
    if os.name=='nt':return executable(value)
    where=shutil.which('where.exe')
    if not where:raise RuntimeError('Windows interop is missing where.exe')
    result=subprocess.run([where,value],capture_output=True,text=True)
    candidates=[line.strip().rstrip('\r') for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or not candidates:
        raise RuntimeError(f'Missing Windows executable {value}; install it or pass --recorder-ffmpeg')
    return candidates[0]

def recorder_ffmpeg(value):
    if not value:return find_windows_executable('ffmpeg.exe')
    if os.name=='nt':return executable(value)
    if ':' in value or '\\' in value:return value
    return find_windows_executable(value)

def powershell_file_command(script_path,*arguments):
    return [executable('powershell.exe'),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
            '-File',windows_path(script_path),*arguments]

def launch_windows_game(command,cwd,out,log):
    """Launch a Windows process from WSL while retaining its real Windows PID."""
    relay_script=out/'launch_windows_game.ps1'
    pid_file=out/'windows_game.pid'
    exit_file=out/'windows_game.exitcode'
    relay_script.write_text("""param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$PidFile,
    [Parameter(Mandatory = $true)][string]$ExitFile,
    [Parameter(Mandatory = $true)][string]$Arguments
)
$ErrorActionPreference = 'Stop'
$Process = Start-Process -FilePath $Executable -WorkingDirectory $WorkingDirectory -ArgumentList $Arguments -WindowStyle Hidden -PassThru
[IO.File]::WriteAllText($PidFile, [string]$Process.Id)
$Process.WaitForExit()
[IO.File]::WriteAllText($ExitFile, [string]$Process.ExitCode)
exit $Process.ExitCode
""",encoding='utf-8')
    relay=powershell_file_command(relay_script,'-Executable',command[0],'-WorkingDirectory',cwd,
                                  '-PidFile',windows_path(pid_file),'-ExitFile',windows_path(exit_file),
                                  '-Arguments',subprocess.list2cmdline(command[1:]))
    return subprocess.Popen(relay,stdout=log,stderr=subprocess.STDOUT),pid_file,relay

def terminate_windows_game(pid_file,expected_config):
    if not pid_file.is_file():return False
    try:pid=int(pid_file.read_text(encoding='utf-8-sig').strip())
    except (OSError,ValueError):return False
    script_path=pid_file.parent/'terminate_windows_game.ps1'
    # Kill only if the PID still belongs to this run's game (its command line names our runtime config).
    script_path.write_text("""param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [Parameter(Mandatory = $true)][string]$ExpectedConfig
)
$ErrorActionPreference = 'Stop'
$Process = Get-CimInstance Win32_Process -Filter ("ProcessId=$ProcessId")
if ($null -eq $Process -or [string]::IsNullOrEmpty($Process.CommandLine) -or -not $Process.CommandLine.Contains($ExpectedConfig)) {
    exit 3
}
& taskkill.exe /PID $ProcessId /T /F | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
""",encoding='utf-8')
    result=subprocess.run(powershell_file_command(script_path,'-ProcessId',str(pid),'-ExpectedConfig',windows_path(expected_config)),
                          capture_output=True,text=True,timeout=20)
    return result.returncode==0

def validate_arguments(p,a):
    if a.max_missing_frames<0:p.error('--max-missing-frames must be non-negative')
    if a.replay_poses and a.max_missing_frames:p.error('--max-missing-frames applies only when inferring')
    if a.postprocess_only and (a.pose_cache or a.replay_poses or a.max_missing_frames):
        p.error('pose cache options cannot be combined with --postprocess-only')
    if a.target_width_scale is not None and (not math.isfinite(a.target_width_scale) or a.target_width_scale<=0):p.error('--target-width-scale must be finite and positive')
    if a.target_height_scale is not None and (not math.isfinite(a.target_height_scale) or a.target_height_scale<=0):p.error('--target-height-scale must be finite and positive')
    if not math.isfinite(a.screen_up_pixels):p.error('--screen-up-pixels must be finite')
    if not math.isfinite(a.screen_up_cm):p.error('--screen-up-cm must be finite')
    if a.packaged_exe and a.postprocess_only:p.error('--packaged-exe cannot be combined with --postprocess-only')
    if a.game_project and not a.packaged_exe:p.error('--game-project requires --packaged-exe pointing to UnrealEditor.exe')
    if a.game_project and Path(a.game_project).suffix.lower()!='.uproject':p.error('--game-project must be a .uproject file')
    if a.joint_diagnostics and not a.packaged_exe:p.error('--joint-diagnostics requires --packaged-exe')
    if a.camera_calibration and not a.auto_camera:p.error('--camera-calibration requires --auto-camera')
    if a.auto_camera and not a.packaged_exe:p.error('--auto-camera requires --packaged-exe')
    if a.auto_camera and (a.screen_up_pixels!=0 or a.screen_up_cm!=0):p.error('--auto-camera requires zero screen-up offsets')
    if a.auto_camera and (a.target_width_scale is not None or a.target_height_scale is not None):p.error('--auto-camera cannot be combined with target scale options')
    if not a.packaged_exe and (a.target_width_scale is not None or a.target_height_scale is not None):p.error("target scale options require --packaged-exe")
    if not a.packaged_exe and (a.screen_up_pixels!=0 or a.screen_up_cm!=0):p.error("screen-up offset options require --packaged-exe")
    if not a.packaged_exe and not a.postprocess_only:
        p.error('--packaged-exe is required for recording; this repository supports packaged recording only (ue/record.py is not included)')

def auto_camera_inputs(a,source_path,source,out,summary):
    """Return the runtime-config camera fields and record how the source intrinsics were validated."""
    camera_runtime={'auto_camera_from_pose':True}
    if a.camera_calibration:
        from camera_calibration import validate_canonical,validate_source_identity
        validated_runtime,camera_report=validate_canonical(source_path,Path(a.camera_calibration).resolve())
        validate_source_identity(camera_report,source)
        camera_runtime.update(validated_runtime)
    else:
        camera_report={'type':'auto_camera_input_validation','version':1,
            'mode':'sat_hmr_pelvis_camera_translation','source':str(source_path),
            'canonical_intrinsics_validated':False,'extrinsics_applied':False,
            'description':'Source-video intrinsics are estimated independently. Camera position comes from SAT-HMR 3D translation after SAT-to-UE unit conversion and is not fitted to source joints.'}
    summary['auto_camera_input_validation']=camera_report
    (out/'auto_camera_input_validation.json').write_text(json.dumps(camera_report,indent=2),encoding='utf-8')
    return camera_runtime

def validate_auto_camera_ready(runtime):
    auto=runtime.get('auto_camera',{})
    if auto.get('enabled') is not True or auto.get('mode')!='sat_hmr_pelvis_camera_translation':
        raise RuntimeError('Recorder does not support pose-driven camera placement')

def finite_numbers(values):
    return all(isinstance(item,(int,float)) and math.isfinite(item) for item in values)

def load_video_observations(path,source_path,source_stat,source):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('type')!='video_camera_observations' or value.get('version')!=1:
        raise RuntimeError('Unsupported video camera observations')
    identity=value.get('source',{})
    expected={'path':str(source_path),'size':source_stat.st_size,'mtime_ns':source_stat.st_mtime_ns,
              'frame_count':source['frames'],'width':source['width'],'height':source['height']}
    if any(identity.get(key)!=expected_value for key,expected_value in expected.items()):
        raise RuntimeError('Video camera observation source identity mismatch')
    if Fraction(identity.get('fps','0'))!=Fraction(source['fps']):
        raise RuntimeError('Video camera observation FPS mismatch')
    intrinsics=value.get('intrinsics')
    if not isinstance(intrinsics,list) or len(intrinsics)!=4 or not finite_numbers(intrinsics) or intrinsics[0]<=0 or intrinsics[1]<=0:
        raise RuntimeError('Video camera observations contain invalid intrinsics')
    frames=value.get('frames')
    if not isinstance(frames,list) or len(frames)!=source['frames']:
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

def wait_child(child,timeout_s,name):
    try:code=child.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:raise TimeoutError(name+' timed out')
    if code:raise RuntimeError(f'{name} exited with code {code}; inspect its log in the output directory')

def raise_if_runtime_error(out):
    failure=out/'runtime_error.json'
    if failure.exists():raise RuntimeError(failure.read_text(encoding='utf-8'))

def prepare_output(out):
    if out.exists():raise FileExistsError('Output directory must be new: '+str(out))
    out.mkdir(parents=True)

def resolve_game(a):
    game=Path(a.packaged_exe).resolve()
    if not a.game_project:
        # Prefer the real game binary over the package-root bootstrap launcher when it is unambiguous.
        nested=game.parent/'MyProject_ROMP'/'Binaries'/'Win64'
        candidates=list(nested.glob('MyProject_ROMP*.exe')) if nested.is_dir() else []
        if len(candidates)==1:game=candidates[0]
    if not game.is_file():raise FileNotFoundError(game)
    project=Path(a.game_project).resolve() if a.game_project else None
    if project and not project.is_file():raise FileNotFoundError(project)
    return game,project

def runtime_config(a,source,raw,inbox,camera_runtime,out):
    """Build the SampleVideoRecorder runtime config; its keys are validated by the packaged game."""
    rate=Fraction(source['fps'])
    config={'output':windows_path(raw),'inbox':windows_path(inbox),
            'ffmpeg':recorder_ffmpeg(a.recorder_ffmpeg),'width':source['width'],
            'height':source['height'],'fps_num':rate.numerator,'fps_den':rate.denominator,
            'repeat':a.repeat,'timeout':a.frame_timeout,'camera_tag':'SAT_HMR_Perspective_FOV60',
            'target_class':'BP_UDPBoneBasedReceiver1_C',
            'scene_mode':a.scene_mode,
            'screen_up_pixels':a.screen_up_pixels,'screen_up_cm':a.screen_up_cm}
    if camera_runtime:config.update(camera_runtime)
    if a.joint_diagnostics:config['joint_diagnostics_path']=windows_path(out/'ue_joints.jsonl')
    if a.target_width_scale is not None or a.target_height_scale is not None:
        scale=[a.target_width_scale or 1.0,1.0,a.target_height_scale or 1.0]
        config['target_component_scales']={name:scale for name in ('SMPLcomp','ProcMesh','clothMesh','ClothMeshCompm')}
    return config

def game_command(game,project,config_path,out):
    command=[windows_path(game)]
    if project:command.extend((windows_path(project),'/Game/NewMap2','-game'))
    else:command.append('/Game/NewMap2')
    command.extend(('-SampleVideoConfig='+windows_path(config_path),
                    '-windowed','-ResX=1280','-ResY=720','-unattended','-nosplash',
                    '-abslog='+windows_path(out/'game.log')))
    return command

def wait_for_recorder_ready(recorder,out):
    deadline=time.monotonic()+RECORDER_STARTUP_TIMEOUT_S
    ready=out/'ready.json'
    while not ready.exists():
        raise_if_runtime_error(out)
        if recorder.poll() is not None:raise RuntimeError('Recorder startup failed; inspect recording.log/game.log')
        if time.monotonic()>deadline:raise TimeoutError('Recorder startup timed out; inspect logs')
        time.sleep(.2)
    return json.loads(ready.read_text(encoding='utf-8'))

def inference_command(a,source_path,source_stat,source,inbox,observations_path):
    identity=['--source-identity',str(source_path),'--source-size',str(source_stat.st_size),
              '--source-mtime-ns',str(source_stat.st_mtime_ns),'--source-frame-count',str(source['frames']),
              '--source-fps',str(source['fps'])]
    if a.replay_poses:
        infer=[sys.executable,str(HERE/'infer_sequential.py'),'--source',str(source_path),'--inbox',str(inbox),
               '--timeout',str(a.frame_timeout),*identity,'--replay-poses',str(Path(a.replay_poses).resolve())]
    else:
        infer=[sys.executable,str(HERE/'infer_sequential.py'),
               '--sat-root',str(Path(a.sat_root).resolve()),
               '--source',str(source_path),'--inbox',str(inbox),'--timeout',str(a.frame_timeout),
               *identity,'--max-missing-frames',str(a.max_missing_frames)]
        if a.pose_cache:infer.extend(('--pose-cache',str(Path(a.pose_cache).resolve())))
    if observations_path:infer.extend(('--observations',str(observations_path)))
    return infer

def wait_for_inference(inference,recorder,out,timeout_s):
    """Wait for inference while echoing its latest acknowledged frame and failing fast on recorder errors."""
    deadline=time.monotonic()+timeout_s;last_progress=''
    while inference.poll() is None:
        raise_if_runtime_error(out)
        if recorder.poll() not in (None,0):raise RuntimeError('Recorder failed before inference completed; inspect logs')
        if time.monotonic()>deadline:raise TimeoutError('Inference pipeline timed out')
        lines=(out/'inference.log').read_text(encoding='utf-8',errors='replace').splitlines()
        progress=next((x for x in reversed(lines) if 'acknowledged_frame' in x),'')
        if progress and progress!=last_progress:print(progress,flush=True);last_progress=progress
        time.sleep(1)

def validate_recording_status(raw,source,summary):
    recording_report=raw.with_suffix('.recording.json')
    if not recording_report.exists():return
    status=json.loads(recording_report.read_text(encoding='utf-8'))
    summary['recording_status']=status
    expected_duration_s=source['frames']/float(Fraction(source['fps']))
    if abs(status['simulation_seconds']-expected_duration_s)>1e-5:raise RuntimeError('Recorded simulation time does not match source frame count/FPS')
    if status['simulation_ticks']!=source['frames']*status['simulation_substeps_per_frame']:raise RuntimeError('Recorded simulation tick count is incorrect')

def stop_recorder(recorder,out,windows_game_pid_file,config_path,summary):
    """Ask the recorder to cancel; force-kill it only if it does not exit in time."""
    (out/'cancel').touch()
    try:recorder.wait(timeout=35);return
    except subprocess.TimeoutExpired:pass
    if os.name=='nt':
        subprocess.run(['taskkill.exe','/PID',str(recorder.pid),'/T','/F'],capture_output=True)
    elif not terminate_windows_game(windows_game_pid_file,config_path):
        summary['recorder_cleanup_error']='Could not validate and terminate the Win64 recorder process'
    try:recorder.wait(timeout=10)
    except subprocess.TimeoutExpired:pass

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packaged-exe',help='Cooked Win64 game executable; does not use Editor/Python remote control')
    p.add_argument('--game-project',help='Run packaged-exe as UnrealEditor -game with this .uproject (development trials only)')
    p.add_argument('--source',required=True)
    p.add_argument('--output-dir');p.add_argument('--postprocess-only',action='store_true');p.add_argument('--recording')
    p.add_argument('--repeat',type=int,default=6)
    p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--ffprobe',default='ffprobe')
    p.add_argument('--recorder-ffmpeg',help='Windows ffmpeg executable used by the Win64 Unreal recorder')
    p.add_argument('--observation-python',default=sys.executable,
                   help='Python used for camera observation (defaults to the active interpreter)')
    p.add_argument('--sat-root',default=str(REPO_ROOT/'sat-hmr'),help='SAT-HMR checkout in this repository')
    p.add_argument('--timeout',type=float,default=7200);p.add_argument('--frame-timeout',type=float,default=600)
    pose_group=p.add_mutually_exclusive_group()
    pose_group.add_argument('--pose-cache',help='Write a complete reusable pose JSONL cache during SAT-HMR inference')
    pose_group.add_argument('--replay-poses',help='Replay a complete pose JSONL cache with Windows Python')
    p.add_argument('--max-missing-frames',type=int,default=0,help='Hold the last valid pose across this many consecutive detection misses')
    p.add_argument('--key-color');p.add_argument('--similarity',type=float,default=.12);p.add_argument('--blend',type=float,default=.04)
    p.add_argument('--despill',action='store_true')
    p.add_argument('--target-width-scale',type=float)
    p.add_argument('--target-height-scale',type=float)
    p.add_argument('--screen-up-pixels',type=float,default=0.0)
    p.add_argument('--screen-up-cm',type=float,default=0.0)
    p.add_argument('--joint-diagnostics',action='store_true',help='Write projected UE bone coordinates to output/ue_joints.jsonl')
    p.add_argument('--auto-camera',action='store_true',help='Estimate source intrinsics, then place the camera from converted 3D pose translation while keeping the CG actor fixed')
    p.add_argument('--scene-mode',choices=('normal','rockdrop','orbit'),default='normal')
    p.add_argument('--camera-calibration',help='Use an explicitly supplied calibration JSON instead of estimating intrinsics (requires --auto-camera)')
    a=p.parse_args(argv)
    validate_arguments(p,a)
    default_output=RESULTS_ROOT/PACKAGE_NAMES[a.scene_mode]/'Recordings'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out=Path(a.output_dir or default_output).resolve()
    summary={'success':False,'arguments':dict(vars(a)),'output_dir':str(out),'started':datetime.now().isoformat()}
    children=[];handles=[];lock_path=None;output_ready=False
    recorder=None;windows_game_pid_file=None;config_path=None
    try:
        lock_path=Path.home()/'sample_video_pipeline.lock'
        try:lock_fd=os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:raise RuntimeError(f'Pipeline lock exists: {lock_path}. Finish the active run; remove it only after confirming no pipeline is running')
        os.write(lock_fd,str(os.getpid()).encode());os.close(lock_fd)
        prepare_output(out);output_ready=True
        from postprocess import probe,ensure_cfr,process
        a.ffmpeg=executable(a.ffmpeg);a.ffprobe=executable(a.ffprobe)
        source=probe(a.source,a.ffprobe);ensure_cfr(a.source,source['fps'],a.ffprobe)
        summary['source']=source
        source_path=Path(a.source).resolve()
        source_stat=source_path.stat()
        camera_runtime=auto_camera_inputs(a,source_path,source,out,summary) if a.auto_camera else None

        observations_path=None
        if a.auto_camera and not a.postprocess_only:
            observations_path=out/'video_camera_observations.json'
            observe=[executable(a.observation_python),str(HERE/'observe_video_camera.py'),
                     '--source',str(source_path),'--output',str(observations_path),'--source-identity',str(source_path),
                     '--source-size',str(source_stat.st_size),'--source-mtime-ns',str(source_stat.st_mtime_ns),
                     '--frame-count',str(source['frames']),'--width',str(source['width']),'--height',str(source['height']),
                     '--fps',str(source['fps'])]
            if 'expected_intrinsics' in camera_runtime:
                observe.extend(('--intrinsics',*(str(value) for value in camera_runtime['expected_intrinsics'])))
            summary['observation_command']=observe
            observation_log=open(out/'observation.log','w',encoding='utf-8');handles.append(observation_log)
            observation=subprocess.Popen(observe,stdout=observation_log,stderr=subprocess.STDOUT)
            children.append(observation);wait_child(observation,a.timeout,'Video camera observation')
            observed=load_video_observations(observations_path,source_path,source_stat,source)
            camera_runtime['expected_intrinsics']=observed['intrinsics']
            summary['video_camera_observations']={key:value for key,value in observed.items() if key!='frames'}

        if a.postprocess_only:
            if not a.recording:raise ValueError('--postprocess-only requires --recording existing.avi')
            raw=Path(a.recording).resolve()
        else:
            raw=out/'raw.avi';inbox=out/'inbox'
            game,project=resolve_game(a)
            inbox.mkdir()
            config_path=out/'runtime_config.json'
            config_path.write_text(json.dumps(runtime_config(a,source,raw,inbox,camera_runtime,out)),encoding='utf-8')
            command=game_command(game,project,config_path,out)
            log=open(out/'recording.log','w',encoding='utf-8');handles.append(log)
            if os.name=='nt':
                recorder=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=game.parent)
            else:
                recorder,windows_game_pid_file,relay_command=launch_windows_game(
                    command,windows_path(game.parent),out,log)
            children.append(recorder)
            summary['runtime']=wait_for_recorder_ready(recorder,out)
            if summary['runtime'].get('scene_mode')!=a.scene_mode:
                raise RuntimeError(f"Runtime scene mode mismatch: requested {a.scene_mode}, got {summary['runtime'].get('scene_mode')}")
            if a.auto_camera:validate_auto_camera_ready(summary['runtime'])
            infer=inference_command(a,source_path,source_stat,source,inbox,observations_path)
            summary['record_command']=command;summary['inference_command']=infer
            if os.name!='nt':summary['record_launcher']=relay_command
            inference_log=open(out/'inference.log','w',encoding='utf-8');handles.append(inference_log)
            inference=subprocess.Popen(infer,stdout=inference_log,stderr=subprocess.STDOUT);children.append(inference)
            wait_for_inference(inference,recorder,out,a.timeout)
            wait_child(inference,1,'Inference');wait_child(recorder,60,'Recorder')

        validate_recording_status(raw,source,summary)
        a.recording=str(raw);a.output_dir=str(out/'composite')
        summary['postprocess']=process(a,source)
        summary['success']=True
        print('Completed: '+str(out/'composite'/'final.mp4'),flush=True)
    except BaseException as exc:
        summary['error']=str(exc)
        print('Pipeline failed: '+str(exc),file=sys.stderr,flush=True)
        raise
    finally:
        if recorder is not None and recorder.poll() is None:
            stop_recorder(recorder,out,windows_game_pid_file,config_path,summary)
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for handle in handles:handle.close()
        if lock_path and lock_path.exists():
            # Delete only our own lock, never another process's existing lock.
            if lock_path.read_text()==str(os.getpid()):lock_path.unlink()
        summary['finished']=datetime.now().isoformat()
        if output_ready:(out/'pipeline_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if __name__=='__main__':main()
