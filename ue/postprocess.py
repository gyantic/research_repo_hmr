"""Video-only postprocess: measured timing, sampled chroma key, silent output."""
import argparse
import collections
import json
from fractions import Fraction
from pathlib import Path
import subprocess


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def canonical_sidecar(path):
    """Return the calibrated canonical.mkv sidecar when it is newer than the video, else None."""
    sidecar=Path(str(path)+'.json')
    if Path(path).name!='canonical.mkv' or not sidecar.is_file():return None
    info=json.loads(sidecar.read_text(encoding='utf-8'))
    if 'canonical_K' not in info or sidecar.stat().st_mtime_ns<Path(path).stat().st_mtime_ns:return None
    return info


def probe(path, ffprobe='ffprobe'):
    info=canonical_sidecar(path)
    if info is not None:
        rate=Fraction(info['fps']).limit_denominator(1001000)
        return {'path':str(path),'duration':info['frames']/float(rate),'fps':str(rate),'frames':info['frames'],'width':info['width'],'height':info['height'],'has_audio':False}
    data=json.loads(run([ffprobe,'-v','error','-count_frames','-show_streams','-show_format','-of','json',path],capture_output=True,text=True).stdout)
    video=next(s for s in data['streams'] if s['codec_type']=='video')
    fps=Fraction(video['avg_frame_rate'])
    if fps<=0: raise ValueError('Invalid video frame rate')
    count=int(video.get('nb_read_frames') or video.get('nb_frames'))
    duration=video.get('duration')
    # Matroska often has no stream duration; decoded CFR frame count is authoritative.
    if duration in (None,'N/A'): duration=count/float(fps)
    result={'path':str(path),'duration':float(duration),'fps':str(fps),'frames':count,
            'width':int(video['width']),'height':int(video['height']),
            'has_audio':any(s['codec_type']=='audio' for s in data['streams'])}
    if abs(result['duration']-count/float(fps))>1.1/float(fps):
        raise ValueError('CFR input required: duration and decoded frame count disagree')
    return result


def ensure_cfr(path, fps, ffprobe):
    info=canonical_sidecar(path)
    if info is not None and abs(info['fps']-float(Fraction(fps)))<1e-5:return
    data=json.loads(run([ffprobe,'-v','error','-select_streams','v:0','-show_frames',
                         '-show_entries','frame=best_effort_timestamp_time','-of','json',path],capture_output=True,text=True).stdout)
    stamps_s=[float(f['best_effort_timestamp_time']) for f in data['frames'] if 'best_effort_timestamp_time' in f]
    step_s=1/float(Fraction(fps))
    if len(stamps_s)<2 or any(abs((b-a)-step_s)>max(.002,step_s*.08) for a,b in zip(stamps_s,stamps_s[1:])):
        raise ValueError('Only CFR source timing is supported; variable frame intervals detected')


def sample_key(path, width, height, ffmpeg):
    # Decode RGB into RAM only. Four small corner patches avoid the foreground.
    raw=run([ffmpeg,'-v','error','-i',path,'-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],capture_output=True).stdout
    if len(raw)!=width*height*3: raise ValueError('Unexpected RGB readback size')
    pixels=[]
    for top,left in [(0,0),(0,width-24),(height-24,0),(height-24,width-24)]:
        for y in range(max(0,top),min(height,top+24)):
            for x in range(max(0,left),min(width,left+24)):
                p=tuple(raw[(y*width+x)*3:(y*width+x)*3+3])
                if p[1]>p[0]+30 and p[1]>p[2]+30: pixels.append(p)
    if len(pixels)<100: raise ValueError('No reliable green border; specify --key-color 0xRRGGBB')
    rgb=collections.Counter(pixels).most_common(1)[0][0]
    return '0x'+''.join(f'{v:02x}' for v in rgb)


def process(args, source=None):
    """Composite args.recording over args.source; pass ``source`` when the caller already ran probe+ensure_cfr on it."""
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    if source is None:
        source=probe(args.source,args.ffprobe)
        ensure_cfr(args.source,source['fps'],args.ffprobe)
    recording=probe(args.recording,args.ffprobe)
    if (source['width'],source['height'])!=(recording['width'],recording['height']): raise ValueError('Source/camera resolution mismatch')
    if Fraction(source['fps'])!=Fraction(recording['fps']): raise ValueError('Source/capture FPS mismatch')
    expected_frames=source['frames']*args.repeat
    if recording['frames']!=expected_frames: raise ValueError(f"Expected {expected_frames} recorded frames, got {recording['frames']}")
    speed_multiplier=recording['duration']/source['duration']
    key=args.key_color or sample_key(args.recording,recording['width'],recording['height'],args.ffmpeg)
    adjusted=output/'speed_adjusted.mkv'; final=output/'final.mp4'
    for path in (adjusted,final,output/'report.json'):
        if path.exists(): raise FileExistsError(path)
    # The recorder emits exactly ``repeat`` consecutive copies for each source
    # frame.  Select by decoded frame index so container duration rounding can
    # never merge frame zero and frame one onto the same integer timestamp.
    rate=Fraction(source['fps'])
    timebase=f'{rate.denominator}/{rate.numerator}'
    timing=f"select=not(mod(n\\,{args.repeat})),settb={timebase},setpts=N"
    run([args.ffmpeg,'-v','warning','-n','-i',args.recording,'-an','-vf',timing,
         '-frames:v',source['frames'],'-vsync','0',
         '-c:v','ffv1','-level','3','-pix_fmt','yuv444p',adjusted])
    # Key on actual compressed video; a restrained feather keeps garment edges.
    key_filter=f'chromakey={key}:{args.similarity}:{args.blend}'
    if args.despill: key_filter+=',despill=type=green:mix=0.5:expand=0'
    graph=f'[0:v]settb={timebase},setpts=N[base];[1:v]settb={timebase},setpts=N,format=yuva444p,{key_filter}[cloth];[base][cloth]overlay=shortest=0:eof_action=repeat:format=auto,format=yuv420p[out]'
    run([args.ffmpeg,'-v','warning','-n','-i',args.source,'-i',adjusted,'-filter_complex',graph,
         '-map','[out]','-an','-c:v','libx264','-crf','18','-preset','medium',
         '-frames:v',source['frames'],'-t',f"{source['duration']:.12g}",
         '-movflags','+faststart',final])
    adjusted_info=probe(adjusted,args.ffprobe); final_info=probe(final,args.ffprobe)
    for info in (adjusted_info,final_info):
        if info['frames']!=source['frames'] or abs(info['duration']-source['duration'])>1.1/float(Fraction(source['fps'])):
            raise ValueError(f'Output timing validation failed: {info}')
    report={'source':source,'recording':recording,'speed_multiplier':speed_multiplier,
            'frame_mapping':{'method':'decoded_frame_index','recording_index':f'source_index * {args.repeat}'},
            'key_color':key,
            'similarity':args.similarity,'blend':args.blend,'despill':args.despill,
            'adjusted':adjusted_info,'final':final_info,'animation_bake':False,'image_sequence':False}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--recording',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--repeat',type=int,default=6)
    p.add_argument('--key-color');p.add_argument('--similarity',type=float,default=.12)
    p.add_argument('--blend',type=float,default=.04);p.add_argument('--despill',action='store_true')
    p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--ffprobe',default='ffprobe')
    args=p.parse_args()
    if not 0.01<=args.similarity<=1 or not 0<=args.blend<=1 or args.repeat<1: p.error('Invalid key/repeat settings')
    process(args)
if __name__=='__main__': main()
