from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .schema import Plan


def render_plan(plan: Plan, path: str, min_label_len: float = 0.6) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = plt.cm.Pastel1
    for r in plan.rooms:
        poly = np.array(r.polygon)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(r.id % 9), alpha=0.85, zorder=1)
        ax.plot(*np.vstack([poly, poly[:1]]).T, color="#222", lw=2.2, zorder=3)
        c = poly.mean(axis=0)
        ch = f"\nceiling {r.ceiling_height.value:.2f} m" if r.ceiling_height else "\nceiling n/a"
        ax.text(*c, f"Room {r.id}\n{r.area.value:.1f} m²" + ch, ha="center", va="center", fontsize=8, zorder=5)
        for w in r.walls:
            if w.length.value < min_label_len:
                continue
            a, b = np.array(w.p0), np.array(w.p1)
            mid, d = (a + b) / 2, b - a
            n = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)       # outward for CCW polygons
            ang = np.degrees(np.arctan2(d[1], d[0]))
            ang = ang + 180 if ang > 90 or ang < -90 else ang
            ax.text(*(mid + n * 0.22), f"{w.length.value:.2f}", fontsize=7, ha="center", va="center",
                    rotation=ang, color="#444", zorder=6)
    for o in plan.openings:
        a, b = np.array(o.p0), np.array(o.p1)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#2a9d4a" if o.kind == "door" else "#e08a00", lw=5,
                solid_capstyle="butt", zorder=4)
        mid = (a + b) / 2
        ax.text(*mid, f"{o.width.value:.2f}", fontsize=6, color="#1a6b30", ha="center", va="bottom", zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"roomscan plan - tier: {plan.tier} - {len(plan.rooms)} rooms (intervals provisional)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)