import socket

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation


class UESender:
    """UDP/TCP 両対応で Unreal Engine に SMPL データを送信するクラス。

    SAT-HMR (github.com/yangtiming/SAT-HMR) の UE 通信プロトコルと互換。

    ペイロード形式:
        "{96 quaternion values}|{3 translation values}<terminator>"
        - UDP: 改行 \\n で終端
        - TCP: NULL \\x00 で終端
        - quaternion: 24 joints × 4 (x,y,z,w)
        - translation: (x,y,z) cm

    使い方::
        sender = UESender(protocol="udp")
        sender.wait_for_connection()   # ブロック: UE 接続/HELLO 待機
        sender.send(global_rot, smpl_pose, pred_cam_t)
        sender.close()
    """

    def __init__(
        self,
        protocol: str = "udp",
        ue_host: str = "0.0.0.0",
        ue_port: int = 62004,
        local_port: int = 5555,
    ):
        """
        Args:
            protocol: "udp" or "tcp"
            ue_host:  UDP 送信先 IP（UDP では HELLO 受信時に自動検出するため通常不要）。
                      TCP ではバインドアドレス（"0.0.0.0" で全 IF 受け付け）。
            ue_port:  UDP 送信先ポート（UE 受信ポート）。TCP では未使用。
            local_port: UDP 待受ポート (HELLO 受信) / TCP リッスンポート。
        """
        if protocol not in ("udp", "tcp"):
            raise ValueError(f"protocol must be 'udp' or 'tcp', got {protocol!r}")

        self.protocol = protocol
        self.ue_host = ue_host
        self.ue_port = ue_port
        self.local_port = local_port

        self.is_connected = False
        self._sock = None
        self._conn = None       # TCP のみ: accept() で得た接続ソケット
        self._udp_target = None  # UDP のみ: (host, port) 送信先

    # ------------------------------------------------------------------
    # 接続確立
    # ------------------------------------------------------------------

    def wait_for_connection(self) -> None:
        """UE との接続が確立するまでブロックする。

        UDP: ローカルポートで HELLO パケットを受信するまで待機。
             送信元アドレスを UE ホストとして保存。
        TCP: ローカルポートで listen し、UE の接続を待つ。
        """
        if self.protocol == "udp":
            self._wait_udp()
        else:
            self._wait_tcp()
        self.is_connected = True

    def _wait_udp(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self.local_port))
        logger.info(f"[UE/UDP] Waiting for HELLO on 0.0.0.0:{self.local_port} ...")
        _, addr = self._sock.recvfrom(2048)
        self._udp_target = (addr[0], self.ue_port)
        logger.info(f"[UE/UDP] HELLO received from {addr} -> will send to {self._udp_target}")

    def _wait_tcp(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.ue_host, self.local_port))
        self._sock.listen(1)
        logger.info(f"[UE/TCP] Listening on {self.ue_host}:{self.local_port} ...")
        self._conn, addr = self._sock.accept()
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info(f"[UE/TCP] UE connected from {addr}")

    # ------------------------------------------------------------------
    # 送信
    # ------------------------------------------------------------------

    def send(
        self,
        global_rot_euler_zyx: np.ndarray,
        smpl_pose_21x3: np.ndarray,
        pred_cam_t: np.ndarray,
    ) -> None:
        """SMPL パラメータを UE 形式に変換して送信する。

        Args:
            global_rot_euler_zyx: (3,) root の Euler ZYX 角（ラジアン）。
                `out["global_rot"]` から取得した値をそのまま渡す。
            smpl_pose_21x3: (21, 3) body joints 1–21 の axis-angle。
                `fusion_runner.infer()` から得られる smpl_pose をそのまま渡す。
            pred_cam_t: (3,) カメラ座標系での root 位置（メートル）。
                `out["pred_cam_t"]` から取得した値をそのまま渡す。
        """
        if not self.is_connected:
            return

        try:
            pose_str, trans_str = self._build_payload(
                global_rot_euler_zyx, smpl_pose_21x3, pred_cam_t
            )
        except Exception as exc:
            logger.warning(f"[UE] payload build failed: {exc}")
            return

        try:
            if self.protocol == "udp":
                payload = f"{pose_str}|{trans_str}\n".encode("ascii")
                self._sock.sendto(payload, self._udp_target)
            else:
                msg = f"{pose_str}|{trans_str}\x00"
                self._conn.sendall(msg.encode("ascii"))
        except Exception as exc:
            logger.warning(f"[UE/{self.protocol.upper()}] send failed: {exc}")

    # ------------------------------------------------------------------
    # ペイロード構築 (座標変換)
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        global_rot_euler_zyx: np.ndarray,
        smpl_pose_21x3: np.ndarray,
        pred_cam_t: np.ndarray,
    ):
        """SMPL パラメータを UE 互換の文字列ペアに変換する。

        Returns:
            (pose_str, trans_str): カンマ区切りの文字列タプル
        """
        # --- Root joint: Euler ZYX → axis-angle (x180 で SMPL 座標系に変換) ---
        x180 = Rotation.from_euler("x", 180.0, degrees=True)
        root_rotvec = (
            x180 * Rotation.from_euler("ZYX", np.asarray(global_rot_euler_zyx, dtype=np.float64))
        ).as_rotvec()

        # --- Joints 0-21: axis-angle (22×3) ---
        pose_22x3 = np.vstack(
            [root_rotvec, np.asarray(smpl_pose_21x3, dtype=np.float64).reshape(21, 3)]
        )

        # axis-angle → quaternion (xyzw)
        quats = Rotation.from_rotvec(pose_22x3).as_quat().astype(np.float32)  # (22,4)

        # UE 座標変換: X,Z 軸反転
        quats[:, 0] *= -1.0
        quats[:, 2] *= -1.0

        # Root joint 補正 (SAT-HMR と同じ flip_root 左乗算)
        flip_root = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # xyzw の単位 X
        quats[0] = (
            Rotation.from_quat(flip_root) * Rotation.from_quat(quats[0])
        ).as_quat().astype(np.float32)

        # Joints 22, 23 (手): identity quaternion
        identity = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        quats_24 = np.vstack([quats, identity])  # (24,4)

        # --- Translation: カメラ空間 (m) → UE 空間 (cm) ---
        trans = np.asarray(pred_cam_t, dtype=np.float32).copy()
        trans[1] *= -1.0   # Y 軸反転 (カメラ→ワールド)
        trans[2] = 0.0     # Z=0 (グラウンドプレーン固定)
        trans *= 100.0     # m → cm

        pose_str = ",".join(f"{v:.4f}" for v in quats_24.flatten())
        trans_str = ",".join(f"{v:.4f}" for v in trans)
        return pose_str, trans_str

    # ------------------------------------------------------------------
    # クリーンアップ
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.is_connected = False
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        logger.info(f"[UE/{self.protocol.upper()}] Closed.")
