"""One command: sequential SAT-HMR -> UE Camera AVI -> timed chroma-key composite."""
import argparse
from datetime import datetime
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from source_checks import load_video_observations

HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parent
PACKAGE_NAMES={'normal':'Normal','rockdrop':'RockDrop','orbit':'SphereOrbit'}
SCENE_MODES=tuple(PACKAGE_NAMES)
RESULTS_ROOT=REPO_ROOT/'results'/'ue'
RECORDER_STARTUP_TIMEOUT_S=180
RECORD_REPEAT=1  # recorded frames per source frame (runtime config and composite must agree)

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

def auto_camera_inputs(source_path,out,summary):
    """Return the runtime-config camera fields and record how the source intrinsics were obtained."""
    camera_report={'type':'auto_camera_input_validation','version':1,
        'mode':'sat_hmr_pelvis_camera_translation','source':str(source_path),
        'canonical_intrinsics_validated':False,'extrinsics_applied':False,
        'description':'Source-video intrinsics are estimated independently. Camera position comes from SAT-HMR 3D translation after SAT-to-UE unit conversion and is not fitted to source joints.'}
    summary['auto_camera_input_validation']=camera_report
    (out/'auto_camera_input_validation.json').write_text(json.dumps(camera_report,indent=2),encoding='utf-8')
    return {'auto_camera_from_pose':True}

def validate_auto_camera_ready(runtime):
    auto=runtime.get('auto_camera',{})
    if auto.get('enabled') is not True or auto.get('mode')!='sat_hmr_pelvis_camera_translation':
        raise RuntimeError('Recorder does not support pose-driven camera placement')

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
    # Prefer the real game binary over the package-root bootstrap launcher when it is unambiguous.
    nested=game.parent/'MyProject_ROMP'/'Binaries'/'Win64'
    candidates=list(nested.glob('MyProject_ROMP*.exe')) if nested.is_dir() else []
    if len(candidates)==1:game=candidates[0]
    if not game.is_file():raise FileNotFoundError(game)
    return game

def runtime_config(a,source,raw,inbox,camera_runtime):
    """Build the SampleVideoRecorder runtime config; its keys are validated by the packaged game."""
    rate=Fraction(source['fps'])
    config={'output':windows_path(raw),'inbox':windows_path(inbox),
            'ffmpeg':recorder_ffmpeg(a.recorder_ffmpeg),'width':source['width'],
            'height':source['height'],'fps_num':rate.numerator,'fps_den':rate.denominator,
            'repeat':RECORD_REPEAT,'timeout':a.frame_timeout,'camera_tag':'SAT_HMR_Perspective_FOV60',
            'target_class':'BP_UDPBoneBasedReceiver1_C',
            'scene_mode':a.scene_mode,
            # No screen-up offset; the packaged game still expects both keys.
            'screen_up_pixels':0.0,'screen_up_cm':0.0}
    if camera_runtime:config.update(camera_runtime)
    return config

def game_command(game,config_path,out):
    return [windows_path(game),'/Game/NewMap2','-SampleVideoConfig='+windows_path(config_path),
            '-windowed','-ResX=1280','-ResY=720','-unattended','-nosplash',
            '-abslog='+windows_path(out/'game.log')]

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

def absolute_path(path):
    return str(Path(path).expanduser().resolve())

def add_common_args(p):
    """Add the options run_video.py exposes and forwards unchanged; return their actions for forward_common_args."""
    actions=[
        p.add_argument('--observation-python',default=sys.executable,
                       help='Python used for camera observation (defaults to the active interpreter)'),
        p.add_argument('--ffmpeg',default='ffmpeg',help='Native ffmpeg for probing and compositing'),
        p.add_argument('--ffprobe',default='ffprobe',help='Native ffprobe for probing and validation'),
        p.add_argument('--recorder-ffmpeg',help='Windows ffmpeg used by the Win64 Unreal package'),
        p.add_argument('--timeout',type=float,default=7200),
        p.add_argument('--frame-timeout',type=float,default=600),
        p.add_argument('--max-missing-frames',type=int,default=0,
                       help='Hold the last valid pose across this many consecutive detection misses')]
    pose_group=p.add_mutually_exclusive_group()
    actions.append(pose_group.add_argument('--pose-cache',type=absolute_path,
                                           help='Write a complete reusable pose JSONL cache during SAT-HMR inference'))
    actions.append(pose_group.add_argument('--replay-poses',type=absolute_path,
                                           help='Replay a complete pose JSONL cache instead of running SAT-HMR'))
    return actions

def forward_common_args(a,actions):
    """Turn the parsed add_common_args options back into command-line arguments for main()."""
    argv=[]
    for action in actions:
        value=getattr(a,action.dest)
        if value is not None:argv.extend((action.option_strings[0],str(value)))
    return argv

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packaged-exe',required=True,help='Cooked Win64 game executable; does not use Editor/Python remote control')
    p.add_argument('--source',required=True)
    p.add_argument('--output-dir')
    p.add_argument('--sat-root',default=str(REPO_ROOT/'sat-hmr'),help='SAT-HMR checkout in this repository')
    add_common_args(p)
    p.add_argument('--despill',action='store_true')
    p.add_argument('--auto-camera',action='store_true',help='Estimate source intrinsics, then place the camera from converted 3D pose translation while keeping the CG actor fixed')
    p.add_argument('--scene-mode',choices=SCENE_MODES,default='normal')
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
        from postprocess import KEY_BLEND,KEY_SIMILARITY,probe,ensure_cfr,process
        a.ffmpeg=executable(a.ffmpeg);a.ffprobe=executable(a.ffprobe)
        source=probe(a.source,a.ffprobe);ensure_cfr(a.source,source['fps'],a.ffprobe)
        summary['source']=source
        source_path=Path(a.source).resolve()
        source_stat=source_path.stat()
        camera_runtime=auto_camera_inputs(source_path,out,summary) if a.auto_camera else None

        observations_path=None
        if a.auto_camera:
            observations_path=out/'video_camera_observations.json'
            observe=[executable(a.observation_python),str(HERE/'observe_video_camera.py'),
                     '--source',str(source_path),'--output',str(observations_path),'--source-identity',str(source_path),
                     '--source-size',str(source_stat.st_size),'--source-mtime-ns',str(source_stat.st_mtime_ns),
                     '--frame-count',str(source['frames']),'--width',str(source['width']),'--height',str(source['height']),
                     '--fps',str(source['fps'])]
            summary['observation_command']=observe
            observation_log=open(out/'observation.log','w',encoding='utf-8');handles.append(observation_log)
            observation=subprocess.Popen(observe,stdout=observation_log,stderr=subprocess.STDOUT)
            children.append(observation);wait_child(observation,a.timeout,'Video camera observation')
            observed=load_video_observations(observations_path,{
                'path':str(source_path),'size':source_stat.st_size,'mtime_ns':source_stat.st_mtime_ns,
                'frame_count':source['frames'],'width':source['width'],'height':source['height'],'fps':source['fps']})
            camera_runtime['expected_intrinsics']=observed['intrinsics']
            summary['video_camera_observations']={key:value for key,value in observed.items() if key!='frames'}

        raw=out/'raw.avi';inbox=out/'inbox'
        game=resolve_game(a)
        inbox.mkdir()
        config_path=out/'runtime_config.json'
        config_path.write_text(json.dumps(runtime_config(a,source,raw,inbox,camera_runtime)),encoding='utf-8')
        command=game_command(game,config_path,out)
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
        composite=argparse.Namespace(source=a.source,recording=str(raw),output_dir=str(out/'composite'),
                                     repeat=RECORD_REPEAT,key_color=None,similarity=KEY_SIMILARITY,blend=KEY_BLEND,
                                     despill=a.despill,ffmpeg=a.ffmpeg,ffprobe=a.ffprobe)
        summary['postprocess']=process(composite,source)
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
