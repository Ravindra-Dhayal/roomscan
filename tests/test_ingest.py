from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from roomscan.ingest import load_capture

DATA = Path(__file__).resolve().parents[1] / "data"
REAL = {"single_room": 1715, "single_scan_floor_only": 5251, "single_scan_with_ceiling": 9745}


def _make_synthetic(root: Path, n=3):
    """Flat wall 2 m in front of an identity-pose camera, full-res RGB 1920x1440."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    np.savetxt(root / "camera_matrix.csv", [[1600, 0, 960], [0, 1600, 720], [0, 0, 1]], delimiter=",")
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    for i in range(n):
        rows.append(f"{i*0.016}, {i:06d}, 0, 0, 0, 0, 0, 0, 1, 1600, 1600, 960, 720, , ")
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.full((192, 256), 2000, np.uint16))
        conf = np.full((192, 256), 2, np.uint8)
        conf[:, :10] = 0  # a low-confidence strip that must be masked out
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), conf)
    (root / "odometry.csv").write_text("\n".join(rows) + "\n")
    (root / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n0,0,-1,0,0,0,0\n")


def test_synthetic_backprojection(tmp_path):
    _make_synthetic(tmp_path / "cap")
    c = load_capture(tmp_path / "cap")
    assert c.n_frames == 3
    assert c.depth_size == (256, 192) and c.rgb_size == (1920, 1440)  
    assert c.K_depth[0, 0] == pytest.approx(1600 * 256 / 1920)
    p = c.points_world(0)
    assert np.allclose(p[:, 2], 2.0)                    
    assert len(p) == 192 * (256 - 10)                   
    assert abs(p[:, 0].mean()) < 0.2                    


def test_nested_extraction_dir(tmp_path):
    _make_synthetic(tmp_path / "outer" / "cap")
    assert load_capture(tmp_path / "outer").n_frames == 3


def test_missing_capture(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_capture(tmp_path)


def _real(name):
    d = DATA / name
    if not d.exists():
        pytest.skip(f"{d} not present")
    return load_capture(d)


@pytest.mark.parametrize("name", REAL)
def test_real_capture_shapes(name):
    c = _real(name)
    assert c.n_frames == REAL[name]
    assert c.depth_size == (256, 192)
    assert c.rgb_size[0] / c.rgb_size[1] == pytest.approx(4 / 3, abs=0.01)
    d = c.depth_m(0)
    assert 0.2 < np.nanmedian(d) < 6.0                 


def test_real_pose_convention_consistent():
    """Depth from two nearby frames must overlap in world space. This is what pinned down
    that poses use OpenCV camera axes: with a y/z flip the overlap collapses to ~7%."""
    c = _real("single_room")
    a, b = c.points_world(100), c.points_world(140)
    d = cKDTree(b).query(a[::10])[0]
    assert np.mean(d < 0.03) > 0.4


def test_real_world_up_axis_is_y():
    """The floor is a sharp peak along world y (not x or z)."""
    c = _real("single_room")
    pts = np.concatenate([c.points_world(i)[::15] for i in range(0, c.n_frames, 15)])
    peaks = []
    for ax in range(3):
        h, _ = np.histogram(pts[:, ax], bins=np.arange(pts[:, ax].min(), pts[:, ax].max(), 0.02))
        peaks.append(h.max() / len(pts))
    assert int(np.argmax(peaks)) == 1 and peaks[1] > 0.05