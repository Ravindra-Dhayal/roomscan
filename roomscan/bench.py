from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import Interval, Plan

# gates from the case study
GATES = {
    "opening_abs_m": 0.02, "opening_frac": 0.85,
    "ceiling_abs_m": 0.015, "ceiling_spread_m": 0.01,
    "repeat_abs_m": 0.01, "repeat_rel": 0.005,
    "wall_rel": {"lidar": None, "video": 0.03, "photo": 0.08},
    "footprint_rel_photo": 0.08,
}
MAX_ASSOC_OPENING = 0.20      # a predicted opening further than 20 cm from every GT width is a phantom


def _covers(iv: Interval, truth: float) -> bool:
    return iv.lo <= truth <= iv.hi


@dataclass
class CaptureResult:
    capture_id: str
    tier: str
    room_match: list[tuple[str, int, float]] = field(default_factory=list)   # (gt_name, pred_id, area_err_rel)
    walls: list[dict] = field(default_factory=list)       # gt_room, truth, pred, err, covered, pred_id, idx
    ceilings: list[dict] = field(default_factory=list)    # gt_room, truth, pred|None, err, covered
    openings: list[dict] = field(default_factory=list)    # kind, truth|None, pred|None, err, hit, covered
    adjacency: dict = field(default_factory=dict)
    footprint_err_rel: float | None = None
    unmatched_gt_rooms: list[str] = field(default_factory=list)
    phantom_pred_rooms: list[int] = field(default_factory=list)


def _match_rooms(plan: Plan, gt: dict):
    gr, pr = gt["rooms"], plan.rooms
    if not gr or not pr:
        return []
    cost = np.array([[abs(p.area.value - (g.get("area") or sum_area(g))) for p in pr] for g in gr])
    i, j = linear_sum_assignment(cost)
    return [(gr[a], pr[b]) for a, b in zip(i, j)]


def sum_area(g: dict) -> float:
    """Fallback GT area: product of the two longest distinct wall lengths (rectangular rooms)."""
    w = sorted(g["walls"], reverse=True)
    return float(w[0] * w[-1]) if w else 0.0


def compare(plan: Plan, gt: dict, capture_id: str = "") -> CaptureResult:
    res = CaptureResult(capture_id or gt.get("capture_id", "?"), plan.tier)
    pairs = _match_rooms(plan, gt)
    name_of_pred: dict[int, str] = {}
    for g, p in pairs:
        gt_area = g.get("area") or sum_area(g)
        res.room_match.append((g["name"], p.id, (p.area.value - gt_area) / gt_area if gt_area else float("nan")))
        name_of_pred[p.id] = g["name"]
        # walls: Hungarian on |length difference|, GT walls drive the match
        pl = [w.length for w in p.walls]
        gl = g["walls"]
        if pl and gl:
            c = np.abs(np.array(gl)[:, None] - np.array([x.value for x in pl])[None, :])
            a, b = linear_sum_assignment(c)
            for ai, bi in zip(a, b):
                res.walls.append(dict(gt_room=g["name"], truth=gl[ai], pred=pl[bi].value,
                                      err=pl[bi].value - gl[ai], covered=_covers(pl[bi], gl[ai]),
                                      pred_id=p.id, idx=int(bi)))
        # ceiling
        gc = g.get("ceiling_height")
        if gc is not None:
            if p.ceiling_height is None:
                res.ceilings.append(dict(gt_room=g["name"], truth=gc, pred=None, err=None, covered=False))
            else:
                res.ceilings.append(dict(gt_room=g["name"], truth=gc, pred=p.ceiling_height.value,
                                         err=p.ceiling_height.value - gc,
                                         covered=_covers(p.ceiling_height, gc)))
    matched_gt = {g["name"] for g, _ in pairs}
    res.unmatched_gt_rooms = [g["name"] for g in gt["rooms"] if g["name"] not in matched_gt]
    res.phantom_pred_rooms = [p.id for p in plan.rooms if p.id not in name_of_pred]

    # openings
    go, po = gt.get("openings", []), plan.openings
    used_p = set()
    if go and po:
        c = np.full((len(go), len(po)), 1e3)
        for a, g in enumerate(go):
            for b, p in enumerate(po):
                if g["kind"] == p.kind:
                    c[a, b] = abs(p.width.value - g["width"])
        a, b = linear_sum_assignment(c)
        used_g = set()
        for ai, bi in zip(a, b):
            if c[ai, bi] <= MAX_ASSOC_OPENING:
                e = po[bi].width.value - go[ai]["width"]
                res.openings.append(dict(kind=go[ai]["kind"], truth=go[ai]["width"], pred=po[bi].width.value,
                                         err=e, hit=abs(e) <= GATES["opening_abs_m"],
                                         covered=_covers(po[bi].width, go[ai]["width"])))
                used_g.add(ai)
                used_p.add(bi)
        for ai, g in enumerate(go):
            if ai not in used_g:
                res.openings.append(dict(kind=g["kind"], truth=g["width"], pred=None, err=None, hit=False, covered=False))
    else:
        for g in go:
            res.openings.append(dict(kind=g["kind"], truth=g["width"], pred=None, err=None, hit=False, covered=False))
    for bi, p in enumerate(po):
        if bi not in used_p:
            res.openings.append(dict(kind=p.kind, truth=None, pred=p.width.value, err=None, hit=False, covered=False))

    # adjacency on matched rooms (by GT name)
    gadj = {tuple(sorted(x)) for x in gt.get("adjacency", [])}
    padj = {tuple(sorted((name_of_pred[a], name_of_pred[b]))) for a, b in plan.adjacency
            if a in name_of_pred and b in name_of_pred}
    tp = len(gadj & padj)
    res.adjacency = dict(truth=len(gadj), pred=len(padj), tp=tp,
                         precision=tp / len(padj) if padj else None, recall=tp / len(gadj) if gadj else None)
    fa = gt.get("footprint_area")
    if fa:
        res.footprint_err_rel = (sum(p.area.value for p in plan.rooms) - fa) / fa
    return res


# ------------------------------------------------------------------------------ gates


def gate_report(results: list[CaptureResult]) -> dict:
    """Aggregate per tier. Returns {tier: {gate_name: {value, gate, passed}}}."""
    out: dict = {}
    for tier in sorted({r.tier for r in results}):
        rs = [r for r in results if r.tier == tier]
        ops = [o for r in rs for o in r.openings]
        rep: dict = {}
        if ops:
            frac = sum(o["hit"] for o in ops) / len(ops)        # misses + phantoms are in the denominator
            rep["opening_width_hit_rate"] = dict(value=frac, gate=GATES["opening_frac"], passed=frac >= GATES["opening_frac"],
                                                 detail=f"{sum(o['hit'] for o in ops)}/{len(ops)} within 2 cm; "
                                                        f"missed={sum(o['pred'] is None for o in ops)} "
                                                        f"phantom={sum(o['truth'] is None for o in ops)}")
        ce = [c for r in rs for c in r.ceilings]
        if ce:
            ok = [c for c in ce if c["err"] is not None and abs(c["err"]) <= GATES["ceiling_abs_m"]]
            rep["ceiling_within_1.5cm"] = dict(value=len(ok) / len(ce), gate=1.0, passed=len(ok) == len(ce),
                                               detail=f"{len(ok)}/{len(ce)} rooms; abstained={sum(c['pred'] is None for c in ce)}")
            errs = [c["err"] for c in ce if c["err"] is not None]
            if errs:
                rep["ceiling_bias_m"] = dict(value=float(np.mean(errs)), gate=None, passed=None,
                                             detail="mean signed error; nonzero + small spread = repeatable-but-biased")
        wl = [w for r in rs for w in r.walls]
        thr = GATES["wall_rel"].get(tier)
        if wl and thr:
            rel = [abs(w["err"]) / w["truth"] for w in wl]
            frac = float(np.mean([x <= thr for x in rel]))
            rep["wall_length_within_tol"] = dict(value=frac, gate=1.0, passed=frac == 1.0,
                                                 detail=f"tolerance {thr*100:.0f}%; max rel err {max(rel)*100:.1f}%")
        elif wl:
            rep["wall_length_mae_m"] = dict(value=float(np.mean([abs(w['err']) for w in wl])), gate=None, passed=None,
                                            detail=f"n={len(wl)} walls; no wall gate for lidar beyond repeatability")
        fp = [r.footprint_err_rel for r in rs if r.footprint_err_rel is not None]
        if fp and tier == "photo":
            m = max(abs(x) for x in fp)
            rep["photo_footprint_within_8pct"] = dict(value=m, gate=GATES["footprint_rel_photo"],
                                                      passed=m <= GATES["footprint_rel_photo"])
        cov = [x["covered"] for r in rs for x in (r.walls + r.ceilings + r.openings) if x.get("truth") is not None]
        if cov:
            rep["interval_coverage_95"] = dict(value=float(np.mean(cov)), gate=0.95, passed=np.mean(cov) >= 0.90,
                                               detail=f"{sum(cov)}/{len(cov)} truths inside their interval (pass >= 90%)")
        out[tier] = rep
    return out


def repeatability(a: CaptureResult, b: CaptureResult) -> dict:
    """Two captures of the same room, same tier: per-wall agreement and ceiling spread."""
    key = lambda w: (w["gt_room"], round(w["truth"], 4))
    wa = {key(w): w["pred"] for w in a.walls}
    rows = []
    for w in b.walls:
        k = key(w)
        if k in wa:
            d = abs(wa[k] - w["pred"])
            L = (wa[k] + w["pred"]) / 2
            rows.append(dict(room=k[0], truth=k[1], diff=d, rel=d / L,
                             passed=(d <= GATES["repeat_abs_m"]) or (d / L <= GATES["repeat_rel"])))
    ca = {c["gt_room"]: c["pred"] for c in a.ceilings if c["pred"] is not None}
    spread = [dict(room=c["gt_room"], spread=abs(ca[c["gt_room"]] - c["pred"]))
              for c in b.ceilings if c["pred"] is not None and c["gt_room"] in ca]
    bias = [c["err"] for c in a.ceilings + b.ceilings if c["err"] is not None]
    diag = None
    if spread and bias:
        sp = max(s["spread"] for s in spread)
        bi = abs(float(np.mean(bias)))
        if sp <= GATES["ceiling_spread_m"] and bi > GATES["ceiling_abs_m"]:
            diag = "repeatable-but-biased"
        elif sp > GATES["ceiling_spread_m"]:
            diag = "unrepeatable"
        else:
            diag = "repeatable and unbiased within gate"
    return dict(walls=rows, ceiling_spread=spread, ceiling_diagnosis=diag,
                walls_passed=all(r["passed"] for r in rows) if rows else None)


def run_manifest(manifest_path: str | Path, run_fn) -> dict:
    """manifest.json: {"captures":[{"id","path","ground_truth","tier"(optional),"repeat_of"(optional)}]}"""
    mp = Path(manifest_path)
    man = json.loads(mp.read_text())
    results: dict[str, CaptureResult] = {}
    timing = {}
    import time
    for c in man["captures"]:
        gt = json.loads((mp.parent / c["ground_truth"]).read_text()) if not Path(c["ground_truth"]).is_absolute() \
            else json.loads(Path(c["ground_truth"]).read_text())
        t = time.perf_counter()
        plan = run_fn(mp.parent / c["path"] if not Path(c["path"]).is_absolute() else c["path"])
        timing[c["id"]] = time.perf_counter() - t
        results[c["id"]] = compare(plan, gt, c["id"])
    reps = {}
    for c in man["captures"]:
        if c.get("repeat_of") and c["repeat_of"] in results:
            reps[f'{c["repeat_of"]}~{c["id"]}'] = repeatability(results[c["repeat_of"]], results[c["id"]])
    return dict(results=results, gates=gate_report(list(results.values())), repeatability=reps, timing_s=timing)


def to_markdown(rep: dict) -> str:
    L = ["# Benchmark report", "", "## Gates", "", "| Tier | Gate | Value | Target | Pass | Detail |", "|---|---|---|---|---|---|"]
    for tier, g in rep["gates"].items():
        for name, v in g.items():
            p = {True: "PASS", False: "FAIL", None: "-"}[v["passed"]]
            tgt = "" if v["gate"] is None else f'{v["gate"]:.3g}'
            L.append(f'| {tier} | {name} | {v["value"]:.4g} | {tgt} | {p} | {v.get("detail", "")} |')
    L += ["", "## Repeatability", ""]
    if not rep["repeatability"]:
        L.append("_No repeated captures in the manifest. The repeatability gate is NOT covered._")
    for k, r in rep["repeatability"].items():
        L.append(f"**{k}**: walls passed = {r['walls_passed']}; ceiling = {r['ceiling_diagnosis']}")
        L += ["", "| room | truth (m) | diff (cm) | rel % | pass |", "|---|---|---|---|---|"]
        for w in r["walls"]:
            L.append(f"| {w['room']} | {w['truth']:.2f} | {w['diff']*100:.1f} | {w['rel']*100:.2f} | {'PASS' if w['passed'] else 'FAIL'} |")
    L += ["", "## Timing (s per capture)", ""] + [f"- {k}: {v:.1f}" for k, v in rep["timing_s"].items()]
    return "\n".join(L) + "\n"