"""Import into sequential SAT-HMR inference; never stores a pose sequence."""
import json
import time
import uuid
from pathlib import Path

def retry_shared_io(operation, timeout=5):
    """Windows/WSL may briefly deny access while publishing shared files."""
    deadline=time.monotonic()+timeout
    while True:
        try:return operation()
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.02)

class PoseRendezvous:
    def __init__(self, inbox, timeout=600):
        self.inbox=Path(inbox);self.timeout=timeout;self.session=uuid.uuid4().hex;self.next_frame=0
        self.inbox.mkdir(parents=True,exist_ok=True)
    def _exchange(self, packet):
        packet=dict(packet,session=self.session)
        request=self.inbox/'request.json';ack=self.inbox/'ack.json'
        if request.exists() or ack.exists(): raise RuntimeError('Rendezvous already has a pending/stale packet')
        temporary=self.inbox/'request.tmp'
        retry_shared_io(lambda: temporary.write_text(json.dumps(packet,allow_nan=False),encoding='utf-8'))
        retry_shared_io(lambda: temporary.replace(request))
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            if ack.exists():
                result=json.loads(retry_shared_io(lambda: ack.read_text(encoding='utf-8')))
                if result.get('session')!=self.session or result.get('frame_index')!=packet.get('frame_index'):
                    raise RuntimeError('ACK does not match this session/frame')
                retry_shared_io(ack.unlink)
                if not result.get('ok'):raise RuntimeError(result)
                return result
            time.sleep(.05)
        raise TimeoutError('UE did not acknowledge; preserve request for diagnosis, do not advance input')
    def submit(self,quaternions,pelvis_camera_m,intrinsics,model_bone_lengths_m=None,model_joints_camera_m=None,
               model_intrinsics=None,body_keypoints_2d=None):
        packet={'frame_index':self.next_frame,'quaternions':quaternions,
                'pelvis_camera_m':pelvis_camera_m,'intrinsics':intrinsics}
        if model_bone_lengths_m is not None:packet['model_bone_lengths_m']=model_bone_lengths_m
        if model_joints_camera_m is not None:packet['model_joints_camera_m']=model_joints_camera_m
        if model_intrinsics is not None:packet['model_intrinsics']=model_intrinsics
        if body_keypoints_2d is not None:packet['body_keypoints_2d']=body_keypoints_2d
        result=self._exchange(packet);self.next_frame+=1;return result
    def finish(self):return self._exchange({'finish':True,'frame_index':self.next_frame})
