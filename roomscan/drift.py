from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from . import lidar
from .ingest import Capture


# ------------------------------------------------------------------------- SE(2) helpers
def _R(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def _compose(a, b):
    """(th, tx, tz) arrays: apply b first, then a."""
    th = a[..., 0] + b[..., 0]
    c, s = np.cos(a[..., 0]), np.sin(a[..., 0])
    tx = c * b[..., 1] - s * b[..., 2] + a[..., 1]
    tz = s * b[..., 1] + c * b[..., 2] + a[..., 2]
    return np.stack([th, tx, tz], -1)


def _inv(a):
    c, s = np.cos(a[..., 0]), np.sin(a[..., 0])
    tx = -(c * a[..., 1] + s * a[..., 2])
    tz = -(-s * a[..., 1] + c * a[..., 2])
    return np.stack([-a[..., 0], tx, tz], -1)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------------------- 2-D ICP
def icp2d(src, dst, iters=40, max_d=(0.30, 0.15, 0.08), trim=0.7):
    """Rigid 2-D ICP, src -> dst. Returns (se2 (th,tx,tz), inlier_frac@5cm, rmse_of_inliers)."""
    tree = cKDTree(dst)
    th, t = 0.0, np.zeros(2)
    cur = src.copy()
    for k in range(iters):
        md = max_d[min(len(max_d) - 1, k * len(max_d) // iters)]
        d, j = tree.query(cur)
        keep = d < md
        if keep.sum() < 30:
            break
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(d[idx])[: max(30, int(trim * len(idx)))]]
        a, b = cur[idx], dst[j[idx]]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        U, _, Vt = np.linalg.svd(H)
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:
            Vt[-1] *= -1
            Ri = Vt.T @ U.T
        ti = cb - Ri @ ca
        cur = cur @ Ri.T + ti
        dth = np.arctan2(Ri[1, 0], Ri[0, 0])
        th += dth
        t = Ri @ t + ti
        if abs(dth) < 1e-5 and np.linalg.norm(ti) < 1e-4:
            break
    d, _ = tree.query(cur)
    inl = d < 0.05
    return np.array([th, t[0], t[1]]), float(inl.mean()), float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else 1.0


# ------------------------------------------------------------------------------- report
@dataclass
class DriftReport:
    enabled: bool = True
    n_nodes: int = 0
    n_candidates: int = 0
    n_loops: int = 0
    max_shift_m: float = 0.0
    max_yaw_deg: float = 0.0
    dy_range_m: float = 0.0
    applied: bool = False
    val_residual_before_m: float = float("nan")
    val_residual_after_m: float = float("nan")
    height_anchor_applied: bool = False
    notes: list[str] = field(default_factory=list)


def solve_graph(n: int, edges: list, sig_seq=(0.003, 0.01, 0.01), sig_loop=(0.004, 0.02, 0.02)) -> np.ndarray:
    """edges: list of (i, j, T, weight) with T = SE(2) mapping cloud_j onto cloud_i. Returns D (n,3)."""
    if not edges:
        return np.zeros((n, 3))
    sig_seq, sig_loop = np.array(sig_seq), np.array(sig_loop)
    ii = np.array([e[0] for e in edges])
    jj = np.array([e[1] for e in edges])
    Ts = np.array([e[2] for e in edges])

    def resid(x):
        Dm = x.reshape(n, 3)
        E = _compose(_inv(Dm[:-1]), Dm[1:])
        E[:, 0] = _wrap(E[:, 0])
        El = _compose(_inv(_compose(Dm[ii], Ts)), Dm[jj])
        El[:, 0] = _wrap(El[:, 0])
        return np.concatenate([Dm[0] / np.array([1e-4, 1e-3, 1e-3]), (E / sig_seq).ravel(), (El / sig_loop).ravel()])

    return least_squares(resid, np.zeros(3 * n), loss="soft_l1", f_scale=3.0, max_nfev=60).x.reshape(n, 3)


def edge_residual(D: np.ndarray, edges: list) -> float:
    """Median loop-closure disagreement (m, with 1 rad ~ 1 m) of the given edges under corrections D."""
    if not edges:
        return float("nan")
    ii = np.array([e[0] for e in edges]); jj = np.array([e[1] for e in edges]); Ts = np.array([e[2] for e in edges])
    E = _compose(_inv(_compose(D[ii], Ts)), D[jj])
    return float(np.median(np.hypot(np.hypot(E[:, 1], E[:, 2]), _wrap(E[:, 0]))))


# ------------------------------------------------------------------------------ the core
def _segments(cap: Capture, seg_len=1.0):
    p = cap.poses[:, [0, 2], 3]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    seg = (arc / seg_len).astype(int)
    _, seg = np.unique(seg, return_inverse=True)
    return seg


def correct(cap: Capture, floor_y: float | None = None, seg_len=1.0, min_gap=6, max_pair_dist=3.0,
            min_inlier=0.55, max_closure_shift=0.5, max_closure_yaw=np.radians(10)) -> tuple[Capture, DriftReport]:
    seg = _segments(cap, seg_len)
    n = int(seg.max()) + 1
    rep = DriftReport(n_nodes=n)
    if n < 2 * min_gap:
        rep.notes.append("trajectory too short for loop closure; no correction applied")
        return cap, rep

    if floor_y is None:
        floor_y = lidar.find_floor_ceiling(lidar.fuse(cap, max_frames=100)).floor.y

    clouds, floor_med, centre = [], np.full(n, np.nan), np.zeros((n, 2))
    for s in range(n):
        fr = np.flatnonzero(seg == s)
        centre[s] = cap.poses[fr][:, [0, 2], 3].mean(axis=0)
        P = np.concatenate([cap.points_world(int(i))[::5] for i in fr[::6]])
        band = P[(P[:, 1] > floor_y + 0.3) & (P[:, 1] < floor_y + 1.8)][:, [0, 2]]
        cells = np.unique(np.floor(band / 0.04).astype(np.int64), axis=0)
        clouds.append((cells + 0.5) * 0.04)
        fl = P[np.abs(P[:, 1] - floor_y) < 0.08, 1]
        if len(fl) > 300:
            floor_med[s] = np.median(fl)

    # candidate loop pairs: far in time, near in space; best few per node
    tree = cKDTree(centre)
    cand = set()
    for j in range(n):
        near = [i for i in tree.query_ball_point(centre[j], max_pair_dist) if j - i >= min_gap]
        for i in sorted(near, key=lambda i: np.linalg.norm(centre[i] - centre[j]))[:4]:
            cand.add((i, j))
    rep.n_candidates = len(cand)
    edges = []
    for i, j in sorted(cand):
        if len(clouds[i]) < 200 or len(clouds[j]) < 200:
            continue
        T, frac, rmse = icp2d(clouds[j], clouds[i])
        if frac >= min_inlier and np.hypot(T[1], T[2]) < max_closure_shift and abs(T[0]) < max_closure_yaw:
            edges.append((i, j, T, frac))
    rep.n_loops = len(edges)

    # Validate before trusting: solve on 2/3 of the loop edges, check that the held-out third
    # agrees better afterwards than with no correction. Otherwise revert to the original poses.
    D = np.zeros((n, 3))
    if len(edges) >= 6:
        val = edges[::3]
        train = [e for k, e in enumerate(edges) if k % 3]
        Dt = solve_graph(n, train)
        before, after = edge_residual(np.zeros((n, 3)), val), edge_residual(Dt, val)
        rep.val_residual_before_m, rep.val_residual_after_m = before, after
        if after < 0.85 * before:
            D = solve_graph(n, edges)
            rep.applied = True
        else:
            rep.notes.append(f"loop correction rejected: held-out closure error {before*100:.1f} -> {after*100:.1f} cm")
    else:
        rep.notes.append(f"loop correction skipped: only {len(edges)} loop closures (need >= 6 to validate)")

    # height anchor
    dy = np.zeros(n)
    ok = np.isfinite(floor_med)
    if ok.sum() >= 3:
        raw = np.where(ok, floor_y - floor_med, np.nan)
        raw = np.interp(np.arange(n), np.flatnonzero(ok), raw[ok])
        k = 5
        pad = np.pad(raw, k // 2, mode="edge")
        dy = np.array([np.median(pad[i:i + k]) for i in range(n)])
    rep.dy_range_m = float(np.ptp(dy))
    if rep.dy_range_m > 0.15:
        rep.notes.append(f"height anchor skipped: floor varies {rep.dy_range_m:.2f} m (steps or multiple levels?)")
        dy = np.zeros(n)
    else:
        rep.height_anchor_applied = bool(ok.sum() >= 3)
    rep.max_shift_m = float(np.max(np.hypot(D[:, 1], D[:, 2])))
    rep.max_yaw_deg = float(np.degrees(np.max(np.abs(D[:, 0]))))

    # interpolate to frames (node k sits at the mean frame index of its segment)
    fidx = np.arange(cap.n_frames)
    nodef = np.array([np.flatnonzero(seg == s).mean() for s in range(n)])
    Df = np.stack([np.interp(fidx, nodef, D[:, k]) for k in range(3)], 1)
    dyf = np.interp(fidx, nodef, dy)
    poses = cap.poses.copy()
    for f in range(cap.n_frames):
        th, tx, tz = Df[f]
        c, s = np.cos(th), np.sin(th)
        Ry = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        poses[f, :3, :3] = Ry @ poses[f, :3, :3]
        poses[f, :3, 3] = Ry @ poses[f, :3, 3] + [tx, dyf[f], tz]
    return replace(cap, poses=poses), rep


# ------------------------------------------------------------------------------ ablation
def quality(pts: np.ndarray, ref: "lidar.SpaceGeometry | None" = None) -> dict:
    """Ground-truth-free sharpness. Wall thickness is measured around the SAME wall lines (taken
    from `ref`, normally the drift-off geometry) in both clouds, so on/off numbers are comparable.
    Lower floor sigma, thinner walls, fewer occupied 2 cm wall cells = less smearing = less drift."""
    g = lidar.analyze(pts)
    ref = ref or g
    fy = ref.vertical.floor.y
    band = pts[(pts[:, 1] > fy + 0.3) & (pts[:, 1] < fy + 1.8)][:, [0, 2]]
    R = lidar._rot(ref.theta)
    q = band @ R
    th, wt = [], []
    for w in ref.walls:
        a = np.array([w.p0, w.p1]) @ R
        axis = 0 if abs(a[0, 0] - a[1, 0]) < abs(a[0, 1] - a[1, 1]) else 1
        c = a[:, axis].mean()
        lo, hi = np.sort(a[:, 1 - axis])
        sel = (np.abs(q[:, axis] - c) < 0.10) & (q[:, 1 - axis] > lo) & (q[:, 1 - axis] < hi)
        if sel.sum() > 100:
            v = q[sel, axis]
            th.append(1.4826 * np.median(np.abs(v - np.median(v))))
            wt.append(sel.sum())
    return dict(floor_sigma_m=g.vertical.floor.sigma,
                wall_thickness_m=float(np.average(th, weights=wt)) if th else float("nan"),
                wall_cells_2cm=int(len(np.unique(np.floor(band / 0.02).astype(np.int64), axis=0))),
                footprint_area_m2=g.footprint.area, geom=g)


def ablation(cap: Capture, out_png: str | None = None) -> dict:
    """Run the pipeline geometry with drift handling OFF and ON; return metrics and (optionally)
    save the side-by-side footprint figure the case study asks for."""
    pts_off = lidar.fuse(cap)
    off = quality(pts_off)
    cap_on, rep = correct(cap, floor_y=off["geom"].vertical.floor.y)
    pts_on = lidar.fuse(cap_on)
    on = quality(pts_on, ref=off["geom"])
    on_geom = lidar.analyze(pts_on)
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
        for a, pts, g, name in ((ax[0], pts_off, off["geom"], "drift handling OFF (poses as-is)"),
                                (ax[1], pts_on, on_geom, "drift handling ON")):
            fy = g.vertical.floor.y
            b = pts[(pts[:, 1] > fy + 0.3) & (pts[:, 1] < fy + 1.8)]
            a.plot(b[::4, 0], b[::4, 2], ",", color="0.55")
            for p in g.footprint.polygons:
                a.plot(*np.vstack([p, p[:1]]).T, "r-", lw=1.6)
            a.set_aspect("equal")
            a.set_title(f"{name}\nfootprint {g.footprint.area:.1f} m2")
        fig.suptitle(f"applied={rep.applied}  loops={rep.n_loops}  floor-height anchor={rep.height_anchor_applied}")
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
    strip = lambda d: {k: v for k, v in d.items() if k != "geom"}
    return dict(off=strip(off), on=strip(on), report=rep)