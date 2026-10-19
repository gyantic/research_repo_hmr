"""One-off: export the first detected ROMP body in VIDEO_PATH as SMPL_proxy.obj.

Saves template.npy (scaled camera-space vertices) and faces_romp.npy, then writes
them as a Wavefront OBJ. Run from the romp/ directory:
    PYTHONPATH="$PWD:$PWD/romp" python romp/make_proxy_obj.py
"""
import pathlib, pickle
import cv2
import numpy as np

from romp.processor_bootstrap import build_video_processor

VIDEO_PATH = "demo/videos/IMG_9732_reencoded.mp4"
VERTEX_SCALE = 300.0     # scale from ROMP camera-space units to proxy mesh units
DEPTH_OFFSET = 500.0     # added to the z (depth) coordinate

romp_proc = build_video_processor("configs/video.yml", VIDEO_PATH, gpu="-1")

pkl_path = pathlib.Path.home() / ".romp" / "smpl_model_data" / "SMPL_NEUTRAL.pkl"
with open(pkl_path, "rb") as f:
    smpl_data = pickle.load(f, encoding="latin1")
faces = np.asarray(smpl_data["f"], dtype=np.int32)

template_saved = False
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    outs = romp_proc.single_image_forward(frame)
    if outs is None or not outs["detection_flag"]:
        continue
    verts = outs["verts_camed"][0].cpu().numpy() * VERTEX_SCALE  # (6890, 3)
    verts[:, 1] *= -1
    verts[:, 2] += DEPTH_OFFSET
    verts[:, [1, 2]] = verts[:, [2, 1]]  # swap the y and z axes

    if not template_saved:
        np.save('template.npy', verts.astype(np.float32))
        np.save('faces_romp.npy', faces)
        print('template.npy と faces_romp.npy を保存しました')
        template_saved = True

v = np.load('template.npy'); f = np.load('faces_romp.npy')
with open('SMPL_proxy.obj','w') as w:
    for x,y,z in v: w.write(f'v {x} {y} {z}\n')
    for a,b,c in f+1: w.write(f'f {a} {b} {c}\n')  # OBJ face indices are 1-based
print('SMPL_proxy.obj を書き出しました')
