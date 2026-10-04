from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


@dataclass
class Capture:
    root: Path
    K_rgb: np.ndarray          
    K_depth: np.ndarray        
    rgb_size: tuple[int, int]  
    depth_size: tuple[int, int]
    timestamps: np.ndarray     
    poses: np.ndarray          
    imu: pd.DataFrame

    @property
    def n_frames(self) -> int:
        return len(self.poses)

    def depth_m(self, i: int, min_conf: int = 1) -> np.ndarray:
        """Depth in metres (H,W); NaN where missing or confidence < min_conf."""
        d = cv2.imread(str(self.root / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"depth frame {i} missing in {self.root}")
        d = d.astype(np.float32) / 1000.0
        d[d <= 0] = np.nan
        c = cv2.imread(str(self.root / "confidence" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if c is not None:
            if c.shape != d.shape:
                c = cv2.resize(c, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
            d[c < min_conf] = np.nan
        return d

    def points_camera(self, i: int, min_conf: int = 1) -> np.ndarray:
        """(M,3) points in the OpenCV camera frame, metres."""
        d = self.depth_m(i, min_conf)
        v, u = np.nonzero(np.isfinite(d))
        z = d[v, u]
        fx, fy = self.K_depth[0, 0], self.K_depth[1, 1]
        cx, cy = self.K_depth[0, 2], self.K_depth[1, 2]
        return np.stack([(u + 0.5 - cx) * z / fx, (v + 0.5 - cy) * z / fy, z], axis=1)

    def points_world(self, i: int, min_conf: int = 1) -> np.ndarray:
        """(M,3) points in the world frame."""
        p = self.points_camera(i, min_conf)
        T = self.poses[i]
        return p @ T[:3, :3].T + T[:3, 3]


def _read_rgb_size(root: Path, K: np.ndarray) -> tuple[int, int]:
    video = root / "rgb.mp4"
    if video.exists():
        cap = cv2.VideoCapture(str(video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    # fallback: principal point is ~ image centre
    return int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))


def load_capture(path: str | Path) -> Capture:
    root = Path(path)
    if not (root / "odometry.csv").exists():
        # tolerate one extra nesting level from zip extraction
        subs = [p for p in root.iterdir() if p.is_dir() and (p / "odometry.csv").exists()]
        if len(subs) != 1:
            raise FileNotFoundError(f"no Stray Scanner capture found under {root}")
        root = subs[0]

    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",").reshape(3, 3)
    odo = pd.read_csv(root / "odometry.csv", skipinitialspace=True)
    quat = odo[["qx", "qy", "qz", "qw"]].to_numpy()
    poses = np.tile(np.eye(4), (len(odo), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    poses[:, :3, 3] = odo[["x", "y", "z"]].to_numpy()

    first = cv2.imread(str(root / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise FileNotFoundError(f"depth/000000.png missing in {root}")
    dh, dw = first.shape[:2]
    rw, rh = _read_rgb_size(root, K)
    S = np.diag([dw / rw, dh / rh, 1.0])
    K_depth = S @ K

    imu = pd.read_csv(root / "imu.csv", skipinitialspace=True)
    return Capture(root, K, K_depth, (rw, rh), (dw, dh),
                   odo["timestamp"].to_numpy(), poses, imu)