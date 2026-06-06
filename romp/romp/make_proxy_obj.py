import cv2, socket, time, sys, pathlib, pickle, json, numpy as np


VIDEO_PATH = "demo/videos/IMG_9732_reencoded.mp4"

arg_list = [
    "--configs_yml", "configs/video.yml",
    "--inputs",      VIDEO_PATH,
    "--gpu",         "-1"            
]

sys.argv = [sys.argv[0]] + arg_list
from romp.lib.config import parse_args, ConfigContext 
from romp.predict.video import Video_processor 

with ConfigContext(parse_args(arg_list)) as cfg:
    romp_proc = Video_processor(args_set=cfg)
    
pkl_path = pathlib.Path.home() / ".romp" / "smpl_model_data" / "SMPL_NEUTRAL.pkl"
with open(pkl_path, "rb") as f:
    smpl_data = pickle.load(f, encoding="latin1")
    
faces = np.asarray(smpl_data["f"], dtype=np.int32)
faces_str = ",".join(map(str, faces.flatten()))
#conn.sendall(f"|{faces_str}\x00".encode("ascii"))
faces_sent = False 
template_saved = False 
dt = 1/15
t_next = time.perf_counter() 
frame_id = 0
"""
faces_str = ",".join(map(str, faces.flatten()))
faces_bin = faces.astype("<i4").tobytes()
msg = b" |" + faces_bin + b"\x00" 
"""

#conn.sendall(msg) 
frame_id = 0
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
            break
    
    outs  = romp_proc.single_image_forward(frame)
    if outs is None or not outs["detection_flag"]:
        frame_id += 1
        continue
    verts = outs["verts_camed"][0].cpu().numpy() * 300.0# (6890,3) float32 #vertsとverts＿camedがある
    verts[:,1] *= -1
    verts[:,2] += 500.0
    verts[:,[1,2]] = verts[:,[2, 1]]
    _, buf = cv2.imencode(".jpg", frame)
        
    if not template_saved:                       # ★ ②
            np.save('template.npy', verts.astype(np.float32))  # ★ ③
            np.save('faces_romp.npy', faces)                  # ★ ④
            print('template.npy と faces_romp.npy を保存しました')
            template_saved = True
v = np.load('template.npy'); f = np.load('faces_romp.npy')
with open('SMPL_proxy.obj','w') as w:
    for x,y,z in v: w.write(f'v {x} {y} {z}\n')
    for a,b,c in f+1: w.write(f'f {a} {b} {c}\n')
print('SMPL_proxy.obj を書き出しました')
