"""Offline camera diagnostics; reads poses/splits, never images or training state.

python -m research.visualize_cameras --snapshot poses.json --out camera-report
python -m research.visualize_cameras --work outputs/prepared/work --case my-scene --out camera-report

Requires numpy, matplotlib and plotly. The output directory is self-contained:
open index.html directly, without a server or internet connection.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys

import numpy as np


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")


def short_name(name):
    group, _, leaf = name.rpartition("/")
    return f"{group}/{leaf[:10]}…" if len(leaf) > 24 else name


def slug(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")


def lpips_status(metrics):
    value = metrics.get("lpips")
    if value is None:
        return {"lpips": None, "points": None, "zero": False, "label": "LPIPS 未知"}
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid LPIPS: {value}")
    points = 40.0 * float(np.clip((0.25 - value) / 0.20, 0, 1))
    return {"lpips": value, "points": points, "zero": points == 0,
            "label": f"LPIPS {value:.4f} · {points:.2f}/40"}


def validate_case(case):
    names = [view["name"] for view in case["views"]]
    if len(names) != len(set(names)) or not names:
        raise ValueError(f"{case['case']}: duplicate/empty views")
    known = set(names)
    split = case["split"]
    for key in ("images", "train", "test", "context"):
        values = split[key]
        if len(values) != len(set(values)) or not set(values) <= known:
            raise ValueError(f"{case['case']}: invalid {key} names")
    if set(split["train"]) & set(split["test"]):
        raise ValueError("train/test overlap")
    if not set(split["context"]) <= set(split["train"]):
        raise ValueError("Context must be a subset of training cameras")
    if set(split["images"]) != set(split["train"]) | set(split["test"]):
        raise ValueError("images must equal train union test")
    if len(split["context"]) < 1:
        raise ValueError("Context is empty; cannot visualize an actual selection")
    for view in case["views"]:
        center = np.asarray(view["center"], dtype=float)
        R = np.asarray(view["R_c2w"], dtype=float)
        if center.shape != (3,) or R.shape != (3, 3) or not np.isfinite(center).all() or not np.isfinite(R).all():
            raise ValueError(f"Invalid camera pose: {view['name']}")
        if not np.allclose(R.T @ R, np.eye(3), atol=2e-4) or not np.isclose(np.linalg.det(R), 1, atol=2e-4):
            raise ValueError(f"R_c2w is not a proper rotation: {view['name']}")
        if min(view["width"], view["height"], view["fx"], view["fy"]) <= 0:
            raise ValueError(f"Invalid intrinsics: {view['name']}")


def estimate_frame(views):
    """Use camera up, with a documented PCA fallback; never assume world XY."""
    centers = np.asarray([view["center"] for view in views])
    rotations = np.asarray([view["R_c2w"] for view in views])
    origin = np.median(centers, axis=0)
    mean_up = -rotations[:, :, 1].mean(axis=0)
    agreement = float(np.linalg.norm(mean_up))
    _, singular, vt = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
    method = "mean_camera_up"
    if agreement > 0.2:
        up = mean_up / agreement
    else:
        # The smallest-variance camera-cloud axis estimates its capture plane.
        up = vt[-1].copy()
        if np.dot(up, -rotations[0, :, 1]) < 0:
            up = -up
        method = "PCA_camera_plane_fallback"
    horizontal = vt[0] - np.dot(vt[0], up) * up
    if np.linalg.norm(horizontal) < 1e-8:
        candidate = np.eye(3)[np.argmin(np.abs(up))]
        horizontal = candidate - np.dot(candidate, up) * up
    horizontal /= np.linalg.norm(horizontal)
    # Fix the arbitrary PCA sign for deterministic screenshots.
    if horizontal[np.argmax(np.abs(horizontal))] < 0:
        horizontal = -horizontal
    second = np.cross(up, horizontal)
    basis = np.column_stack((horizontal, second, up))
    return origin, basis, {"method": method, "mean_up_agreement": agreement,
                           "world_origin": origin.tolist(), "world_to_display_basis": basis.tolist(),
                           "camera_cloud_singular_values": singular.tolist(),
                           "note": "Estimated camera-up plane, not measured gravity; PCA fallback can be ambiguous for a line-like path."}


def policy_names(case):
    """Reuse the project's selector exactly, using only training camera metadata."""
    from research import select_views as selector
    train = sorted(case["split"]["train"])
    k = min(16, len(train))
    by_name = {v["name"]: v for v in case["views"]}
    views = []
    for index, name in enumerate(train):
        v = by_name[name]
        R = np.asarray(v["R_c2w"])
        C = np.asarray(v["center"])
        camera = selector.Camera(index, "PINHOLE", v["width"], v["height"],
                                 v["fx"], v["fy"], v["cx"], v["cy"])
        views.append(selector.View(name, camera, R.T, -R.T @ C, C, R))
    centers = np.asarray([v.center for v in views])
    distance = selector._pairwise(centers)
    errors = {}
    policies = {"actual": list(case["split"]["context"])}
    supplied = case.get("policies", {})
    for policy in ("uniform", "fps", "coverage"):
        if policy in supplied:
            names = supplied[policy]
            if isinstance(names, dict):
                names = names["context"]
            policies[policy] = list(names)
            continue
        try:
            if policy == "uniform":
                order = selector._uniform_order(len(train), k)
            else:
                scale = selector._scene_scale(distance)
                if policy == "fps":
                    order = selector._fps_order(centers, distance, scale, k)[0]
                else:
                    order = selector._coverage_order(views, centers, scale, k)[0]
            policies[policy] = [train[i] for i in sorted(order)]
        except ValueError as error:
            errors[policy] = str(error)
    for policy, names in policies.items():
        if len(names) != len(set(names)) or not set(names) <= set(train):
            raise ValueError(f"Invalid {policy} selection")
    return policies, errors


def geometry_diagnostics(case, names):
    """Distances are scaled by median nonzero training-camera pair distance."""
    by_name = {v["name"]: v for v in case["views"]}
    train = case["split"]["train"]
    centers = np.asarray([by_name[n]["center"] for n in train])
    selected = np.asarray([by_name[n]["center"] for n in names])
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    positive = distance[distance > 1e-10]
    scale = float(np.median(positive)) if positive.size else 1.0
    nearest = np.min(np.linalg.norm(centers[:, None] - selected[None, :], axis=-1), axis=1) / scale
    pair = np.linalg.norm(selected[:, None] - selected[None, :], axis=-1) / scale
    upper = pair[np.triu_indices(len(names), 1)]
    rotations = np.asarray([by_name[n]["R_c2w"] for n in names])
    forward = rotations[:, :, 2]
    turns = np.degrees(np.arccos(np.clip(np.sum(forward[1:] * forward[:-1], axis=1), -1, 1)))
    all_names = list(case["split"]["images"])
    indices = np.array([all_names.index(name) for name in names])
    gaps = np.diff(np.sort(indices))
    training_R = np.asarray([by_name[n]["R_c2w"] for n in train])
    angle = np.degrees(np.arccos(np.clip(training_R[:, :, 2] @ forward.T, -1, 1)))
    orientation_nearest = angle.min(axis=1)
    pivot_index = 8 if len(names) > 8 else len(names) // 2
    return {"count": len(names), "scene_scale": scale,
            "max_train_to_context_distance_normalized": float(nearest.max()),
            "p95_train_to_context_distance_normalized": float(np.quantile(nearest, .95)),
            "min_context_pair_distance_normalized": float(upper.min()) if upper.size else None,
            "max_train_to_context_axis_angle_deg": float(orientation_nearest.max()),
            "max_adjacent_context_axis_turn_deg": float(turns.max()) if turns.size else 0.0,
            "max_filename_order_gap": int(gaps.max()) if gaps.size else 0,
            "filename_indices": indices.tolist(), "pivot_index": pivot_index,
            "pivot_name": names[pivot_index], "names": list(names),
            "note": "Pose-only descriptive diagnostics; center/axis proximity is not surface overlap or perceptual quality. Order is filename/split order, not verified capture time."}


def prepare(case):
    validate_case(case)
    viewmap = {v["name"]: v for v in case["views"]}
    views = [viewmap[n] for n in case["split"]["images"]]
    origin, basis, frame = estimate_frame(views)
    centers = (np.asarray([v["center"] for v in views]) - origin) @ basis
    axes = np.asarray([np.asarray(v["R_c2w"])[:, 2] @ basis for v in views])
    policies, errors = policy_names(case)
    diagnostics = {policy: geometry_diagnostics(case, names) for policy, names in policies.items()}
    return {"case": case, "names": [v["name"] for v in views], "views": views,
            "centers": centers, "axes": axes, "origin": origin, "basis": basis,
            "frame": frame, "policies": policies, "policy_errors": errors,
            "diagnostics": diagnostics, "status": lpips_status(case.get("metrics", {}))}


def draw_topdown(ax, prepared, compact=False):
    case = prepared["case"]
    centers, axes = prepared["centers"], prepared["axes"]
    indices = {name: i for i, name in enumerate(prepared["names"])}
    ax.plot(centers[:, 0], centers[:, 1], c="#a8b1bd", lw=.65, alpha=.6, zorder=1)
    for key, color, marker, label in (("train", "#9ca9b9", ".", "train"),
                                      ("test", "#287dd4", "x", "test")):
        ix = [indices[name] for name in case["split"][key]]
        ax.scatter(centers[ix, 0], centers[ix, 1], c=color, marker=marker,
                   s=12 if compact else 22, label=label, alpha=.8, zorder=2)
    selected = [indices[n] for n in case["split"]["context"]]
    scale = max(float(np.ptp(centers, axis=0).max()), 1e-6)
    ax.quiver(centers[selected, 0], centers[selected, 1], axes[selected, 0], axes[selected, 1],
              color="#bd661c", angles="xy", scale_units="xy", scale=10 / scale,
              width=.004, alpha=.9, zorder=3)
    ax.scatter(centers[selected, 0], centers[selected, 1], s=27 if compact else 45,
               c="#e88727", edgecolors="white", lw=.5, label="context", zorder=4)
    pivot = prepared["diagnostics"]["actual"]["pivot_index"]
    p = selected[pivot]
    ax.scatter(*centers[p, :2], c="#8c37ae", marker="*", s=110 if compact else 180,
               edgecolors="white", lw=.5, label=f"pivot [{pivot}]", zorder=5)
    for number, i in enumerate(selected):
        ax.annotate(str(number), centers[i, :2], xytext=(3, 4), textcoords="offset points",
                    fontsize=5.5 if compact else 8, color="#8a470e", zorder=6)
    status = prepared["status"]
    lp = "unknown" if status["lpips"] is None else f"{status['lpips']:.4f} / {status['points']:.1f} pts"
    reconstructed = case["source"].get("split_status") == "reconstructed"
    title = f"{short_name(case['case'])}\nLPIPS {lp}" + ("  [ZERO SCORE]" if status["zero"] else "")
    ax.set_title(title, color="#c33739" if status["zero"] else "#172c46", fontsize=8 if compact else 13, loc="left")
    for spine in ax.spines.values():
        spine.set_color("#d83e42" if status["zero"] else "#cbd4df")
        spine.set_linewidth(2 if status["zero"] else .8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.14)
    ax.grid(alpha=.15)
    ax.tick_params(labelsize=6 if compact else 9)
    if not compact:
        ax.set_xlabel("estimated plane X (scene units)")
        ax.set_ylabel("estimated plane Y (scene units)")
        ax.legend(loc="best", fontsize=8)
        ax.text(0, -.16, "filename order ≠ verified time | " + ("reconstructed split / default context" if reconstructed else "frozen split / recorded context"),
                transform=ax.transAxes, fontsize=8, color="#596778")
    else:
        ax.text(.01, .015, "RECONSTRUCTED" if reconstructed else "FROZEN SPLIT", transform=ax.transAxes,
                fontsize=5.6, color="#65758b", bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})


def save_pngs(prepared_cases, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out / "topdown").mkdir(parents=True, exist_ok=True)
    for prepared in prepared_cases:
        fig, ax = plt.subplots(figsize=(10, 8))
        draw_topdown(ax, prepared)
        fig.subplots_adjust(bottom=.15, top=.88)
        fig.savefig(out / "topdown" / (slug(prepared["case"]["case"]) + ".png"), dpi=170)
        plt.close(fig)
    columns = 4
    rows = (len(prepared_cases) + columns - 1) // columns
    fig, axs = plt.subplots(rows, columns, figsize=(17, 4.1 * rows), squeeze=False)
    for ax, prepared in zip(axs.flat, prepared_cases):
        draw_topdown(ax, prepared, compact=True)
    for ax in list(axs.flat)[len(prepared_cases):]:
        ax.axis("off")
    fig.suptitle("Camera trajectories · 13-case pose audit\nRed = LPIPS score 0 (LPIPS ≥ 0.25); orange = context, blue = test, purple star = pivot",
                 fontsize=16, color="#172c46")
    fig.text(.04, .015, "Estimated camera-up plane, not measured gravity. Path connects filename order, not verified capture time. Metrics are historical heterogeneous runs; missing values remain unknown.",
             fontsize=9, color="#53647a")
    fig.subplots_adjust(top=.92, bottom=.055, hspace=.34, wspace=.28)
    fig.savefig(out / "topdown_overview.png", dpi=160)
    plt.close(fig)


def segments_trace(points, name, color, width=2, visible=True):
    import plotly.graph_objects as go
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return go.Scatter3d(x=points[:, 0].tolist(), y=points[:, 1].tolist(), z=points[:, 2].tolist(),
                        mode="lines", name=name, line={"color": color, "width": width},
                        visible=visible, hoverinfo="skip")


def frustum_segments(prepared, selected, length):
    points = []
    for i in selected:
        v = prepared["views"][i]
        uv = np.array([[0, 0], [v["width"], 0], [v["width"], v["height"]], [0, v["height"]]])
        local = np.column_stack(((uv[:, 0] - v["cx"]) / v["fx"],
                                 (uv[:, 1] - v["cy"]) / v["fy"], np.ones(4))) * length
        world = local @ np.asarray(v["R_c2w"]).T + np.asarray(v["center"])
        display = (world - prepared["origin"]) @ prepared["basis"]
        center = prepared["centers"][i]
        for corner in display:
            points.extend([center, corner, [np.nan] * 3])
        for j in range(4):
            points.extend([display[j], display[(j + 1) % 4], [np.nan] * 3])
    return points


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f2f5fa;color:#18304a;font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}a{color:#186dbe}button,select{font:inherit;cursor:pointer}button{border:1px solid #c4d0df;border-radius:7px;background:#fff;padding:6px 11px;color:#23415d}header{padding:18px 24px;background:white;border-bottom:1px solid #dce3ed}h1{font-size:21px;margin:0 0 5px}p{margin:5px 0}.muted{color:#61738b;font-size:12px}.bad{color:#bb303b;background:#fff0f0;border:1px solid #f1bdc1}.badge{display:inline-block;border-radius:5px;padding:2px 7px;font-size:12px}.unknown{color:#6e6027;background:#faf5db}.ok{color:#236846;background:#e6f5ed}.notice{padding:9px 12px;border-radius:7px;background:#fff6db;color:#796020;font-size:12px}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 20px;background:white}.plot{height:65vh;min-height:430px}.details{padding:12px 22px;background:white}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #e1e7ee;text-align:left;padding:7px}th{color:#62758c}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;max-height:260px;overflow:auto}.selected{font-weight:700;color:#b45a12}details{margin-top:10px}summary{cursor:pointer;color:#56718d}#index-layout{display:grid;grid-template-columns:325px minmax(0,1fr);height:calc(100vh - 123px)}#cases{padding:14px;overflow:auto}.card{display:block;width:100%;text-align:left;border:1px solid #d5dfeb;border-radius:10px;background:white;margin:0 0 10px;padding:10px;overflow:hidden}.card.active{outline:2px solid #4489d0}.card.bad{border:2px solid #de6970;background:#fff8f8}.card img{float:right;width:83px;height:65px;object-fit:cover;border-radius:5px;margin-left:8px}.card .name{display:block;font-size:13px;overflow-wrap:anywhere;margin-bottom:4px}.card small{display:block;color:#697b8f;font-size:11px}.viewer{min-width:0}iframe{width:100%;height:100%;border:0;background:#fff}@media(max-width:850px){#index-layout{grid-template-columns:220px 1fr}.card img{display:none}header{padding:12px}h1{font-size:17px}}
"""


def scene_html(prepared, out):
    import plotly.graph_objects as go
    case = prepared["case"]
    centers, axes = prepared["centers"], prepared["axes"]
    indices = {name: i for i, name in enumerate(prepared["names"])}
    length = max(float(np.ptp(centers, axis=0).max()), 1e-6) * .045
    traces = []
    labels = []
    for name in prepared["names"]:
        v = prepared["views"][indices[name]]
        labels.append(f"{html.escape(name)}<br>filename index {indices[name]}<br>{v['width']}×{v['height']} | fx={v['fx']:.1f}, fy={v['fy']:.1f}")
    for key, color, symbol in (("train", "#8c9db3", "circle"), ("test", "#2585de", "cross")):
        ix = [indices[n] for n in case["split"][key]]
        traces.append(go.Scatter3d(x=centers[ix, 0], y=centers[ix, 1], z=centers[ix, 2],
                                   mode="markers", name=f"{key} ({len(ix)})", marker={"size": 3, "color": color, "symbol": symbol},
                                   text=[labels[i] for i in ix], hovertemplate="%{text}<extra></extra>"))
    traces.append(segments_trace(centers, "文件名顺序连线（非确认时间）", "#b5c0d0", 2))
    segments = np.stack((centers, centers + axes * length, np.full_like(centers, np.nan)), axis=1).reshape(-1, 3)
    traces.append(segments_trace(segments, "全部 optical axes", "#8eafc1", 2, "legendonly"))
    groups = {}
    for policy, names in prepared["policies"].items():
        ix = [indices[n] for n in names]
        start = len(traces)
        vis = policy == "actual"
        traces.append(go.Scatter3d(x=centers[ix, 0], y=centers[ix, 1], z=centers[ix, 2],
                                   mode="markers+text", name="context + 0-based index", visible=vis,
                                   marker={"size": 6, "color": "#ed902c", "line": {"color": "white", "width": 1}},
                                   text=[str(i) for i in range(len(ix))], textposition="top center",
                                   customdata=[labels[i] for i in ix], hovertemplate="context %{text}<br>%{customdata}<extra></extra>"))
        segments = np.stack((centers[ix], centers[ix] + axes[ix] * length * 1.7,
                             np.full_like(centers[ix], np.nan)), axis=1).reshape(-1, 3)
        traces.append(segments_trace(segments, "context optical axes", "#a95c15", 4, vis))
        # Frusta retain actual FOV/aspect ratio; their depth is display-only.
        traces.append(segments_trace(frustum_segments(prepared, ix, length), "context frusta (display depth)", "#e6b96b", 2, vis))
        pivot = prepared["diagnostics"][policy]["pivot_index"]
        p = ix[pivot]
        traces.append(go.Scatter3d(x=[centers[p, 0]], y=[centers[p, 1]], z=[centers[p, 2]],
                                   mode="markers", name=f"pivot context[{pivot}]", visible=vis,
                                   marker={"size": 10, "color": "#9339ab", "symbol": "diamond"},
                                   text=[labels[p]], hovertemplate="%{text}<extra></extra>"))
        groups[policy] = list(range(start, len(traces)))
    fig = go.Figure(traces)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="white",
                      legend={"x": .01, "y": 1, "font": {"size": 11}, "bgcolor": "rgba(255,255,255,.8)"},
                      scene={"aspectmode": "data", "xaxis_title": "plane X", "yaxis_title": "plane Y", "zaxis_title": "estimated up",
                             "camera": {"up": {"x": 0, "y": 0, "z": 1}, "eye": {"x": 1.5, "y": 1.5, "z": 1.1}}},
                      uirevision="retain-camera")
    status = prepared["status"]
    badge = "bad" if status["zero"] else "unknown" if status["lpips"] is None else "ok"
    warning = "重建划分 / 默认 context，非原实验 context 凭证。" if case["source"].get("split_status") == "reconstructed" else "冻结 split / 已记录 context。"
    rows = []
    for policy, d in prepared["diagnostics"].items():
        rows.append(f"<tr id='row-{policy}'><td>{policy}</td><td>{d['max_train_to_context_distance_normalized']:.3f}</td>"
                    f"<td>{d['min_context_pair_distance_normalized'] if d['min_context_pair_distance_normalized'] is None else format(d['min_context_pair_distance_normalized'], '.3f')}</td>"
                    f"<td>{d['max_train_to_context_axis_angle_deg']:.1f}°</td><td>{d['max_adjacent_context_axis_turn_deg']:.1f}°</td>"
                    f"<td>{d['max_filename_order_gap']}</td><td>{html.escape(d['pivot_name'])}</td></tr>")
    options = "".join(f"<option value='{p}'>{'记录/重建 context' if p == 'actual' else p}</option>" for p in groups)
    provenance = {"source": case["source"], "metrics": case.get("metrics"), "frame": prepared["frame"],
                  "selection_errors": prepared["policy_errors"], "selected_contexts": prepared["policies"]}
    html_body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(case['case'])}</title><style>{STYLE}</style>
<script src="../assets/plotly.min.js"></script><header><h1>{html.escape(short_name(case['case']))} <span class="badge {badge}">{html.escape(status['label'])}{' · 得分为 0' if status['zero'] else ''}</span></h1><p class="muted">{html.escape(case['case'])}</p><div class="notice">{warning} 位姿是真实数据；指标来自历史不同运行，不能视作同条件 13-case 排行榜。</div></header>
<div class="toolbar"><label>Context 策略 <select id="policy">{options}</select></label><button onclick="view('iso')">3D</button><button onclick="view('top')">俯视</button><button onclick="view('front')">侧视</button><a href="../topdown/{slug(case['case'])}.png" target="_blank">俯视 PNG</a><span class="muted">拖动旋转 · 滚轮缩放 · 点图例开关 · 双击复位</span></div><div id="plot" class="plot"></div>
<div class="details"><p class="muted">相机坐标为 COLMAP/OpenCV：+Z 朝前、+Y 朝下。显示坐标由平均 camera-up 估计，弱一致性时用 PCA 平面；不等于重力真值。视锥长度仅用于显示。连线和编号是文件名 / split 顺序，不证明实际采集时间。</p>
<table><thead><tr><th>策略</th><th>最大位置空缺 / 场景尺度 ↓</th><th>最小 context 间距 / 尺度</th><th>最大朝向空缺 ↓</th><th>相邻朝向最大转角</th><th>最大索引间隔</th><th>pivot</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="muted">位置空缺：每个 train 相机到最近 context 中心的距离取最大值；朝向空缺：每个 train 光轴与最近 context 光轴夹角取最大值（位置和朝向分别匹配）。尺度为 train 非零两两距离中位数。它们是几何描述，不等于可见表面覆盖或 LPIPS。coverage 策略是相机包围盒的视锥代理，未考虑遮挡。</p>
<details><summary>来源、坐标估计与逐策略 context 名单</summary><pre>{html.escape(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre></details></div>
<script>const spec={fig.to_json()}, groups={json_text(groups)};
Plotly.newPlot('plot', spec.data, spec.layout, {{responsive:true,displaylogo:false}});
function pick(){{const policy=document.getElementById('policy').value;const all=Object.values(groups).flat();Plotly.restyle('plot',{{visible:all.map(i=>groups[policy].includes(i))}},all);document.querySelectorAll('tbody tr').forEach(r=>r.classList.toggle('selected',r.id==='row-'+policy));}}
document.getElementById('policy').addEventListener('change',pick);document.getElementById('row-actual').classList.add('selected');
function view(which){{const eyes={{iso:{{x:1.5,y:1.5,z:1.1}},top:{{x:0,y:0,z:2.5}},front:{{x:0,y:-2.5,z:.01}}}};Plotly.relayout('plot',{{'scene.camera':{{eye:eyes[which],up:which==='top'?{{x:0,y:1,z:0}}:{{x:0,y:0,z:1}}}}}});}}
</script></html>"""
    path = out / "scenes" / (slug(case["case"]) + ".html")
    path.write_text(html_body, encoding="utf-8")


def write_index(prepared_cases, out, captured_at):
    cases = []
    for prepared in sorted(prepared_cases, key=lambda p: (not p["status"]["zero"], p["case"]["case"])):
        case, status = prepared["case"], prepared["status"]
        cases.append({"case": case["case"], "slug": slug(case["case"]), "name": short_name(case["case"]),
                      "status": status, "split_status": case["source"].get("split_status", "unknown"),
                      "count": len(case["split"]["images"]), "train": len(case["split"]["train"]), "test": len(case["split"]["test"])})
    count = sum(c["status"]["zero"] for c in cases)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FFSplat · 13-case 相机轨迹</title><style>{STYLE}</style>
<header><h1>FFSplat 相机轨迹 · {len(cases)} cases <span class="badge bad">{count} 个 LPIPS 得分为 0</span></h1><p>红色优先排查：LPIPS ≥ 0.25 → 该项 0/40 分。实际 LPIPS = 0 表示高度相似。<a href="topdown_overview.png" target="_blank">全部俯视图</a> · <a href="camera_diagnostics.json" target="_blank">几何诊断 JSON</a></p><p class="muted">离线本地查看 · 一次加载一个 3D 场景 · 历史异条件指标，仅作问题定位 · 元数据 {html.escape(str(captured_at))}</p></header>
<div id="index-layout"><nav id="cases"></nav><main class="viewer"><iframe id="viewer" title="3D camera trajectory"></iframe></main></div>
<script>const cases={json_text(cases)};const nav=document.getElementById('cases'), viewer=document.getElementById('viewer');
function show(c){{viewer.src='scenes/'+c.slug+'.html';document.querySelectorAll('.card').forEach(el=>el.classList.toggle('active',el.dataset.case===c.case));location.hash=encodeURIComponent(c.case);}}
cases.forEach(c=>{{const button=document.createElement('button');button.className='card'+(c.status.zero?' bad':'');button.dataset.case=c.case;const img=document.createElement('img');img.src='topdown/'+c.slug+'.png';img.alt='camera top view';button.appendChild(img);const title=document.createElement('strong');title.className='name';title.textContent=c.name;button.appendChild(title);const badge=document.createElement('span');badge.className='badge '+(c.status.zero?'bad':c.status.lpips===null?'unknown':'ok');badge.textContent=c.status.label+(c.status.zero?' · 零分':'');button.appendChild(badge);const meta=document.createElement('small');meta.textContent=c.train+' train / '+c.test+' test · '+(c.split_status==='frozen'?'冻结划分':'重建划分 / 默认 context');button.appendChild(meta);button.onclick=()=>show(c);nav.appendChild(button);}});
let requested;try{{requested=decodeURIComponent(location.hash.slice(1));}}catch(e){{requested='';}}show(cases.find(c=>c.case===requested)||cases[0]);
</script></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def work_snapshot(work, case_name, lpips):
    from research.select_views import load_scene
    split, by_name = load_scene(work)
    views = []
    for name in split["images"]:
        v = by_name[name]
        camera = v.camera
        views.append({"name": name, "center": v.center.tolist(), "R_c2w": v.R_c2w.tolist(),
                      **{key: getattr(camera, key) for key in ("width", "height", "fx", "fy", "cx", "cy")}})
    hashes = {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (work / "split.json", work / "sparse/0/cameras.txt", work / "sparse/0/images.txt")}
    return {"schema_version": 1, "captured_at": "local prepared work", "cases": [{"case": case_name,
            "source": {"kind": "prepared_COLMAP_text", "path": str(work), "split_status": "frozen", "files": hashes},
            "split": split, "views": views, "metrics": {"lpips": lpips, "source": "CLI --lpips (diagnostic only)"}}]}


def generate(snapshot, out):
    from plotly.offline import get_plotlyjs
    if snapshot.get("schema_version") != 1:
        raise ValueError("Expected snapshot schema_version = 1")
    cases = snapshot["cases"]
    if not cases or len({c["case"] for c in cases}) != len(cases):
        raise ValueError("Empty or duplicate cases")
    prepared = [prepare(case) for case in cases]
    out.mkdir(parents=True, exist_ok=True)
    # Static files deliberately finish first, as requested.
    save_pngs(prepared, out)
    print(f"Saved {len(cases)} top-down PNGs and overview: {out}", flush=True)
    (out / "assets").mkdir(exist_ok=True)
    (out / "scenes").mkdir(exist_ok=True)
    (out / "assets/plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    diagnostics = {"schema_version": 1, "captured_at": snapshot.get("captured_at"),
                   "score_formula": "40 * clip((0.25 - LPIPS) / 0.20, 0, 1)",
                   "notes": ["Historical heterogeneous metrics, not a matched 13-case leaderboard.",
                             "Context policies use only training pose metadata. Test poses are display-only.",
                             "Trajectory order is filename/split order, not verified timestamps."],
                   "cases": [{"case": p["case"]["case"], "source": p["case"]["source"],
                              "metrics": p["case"].get("metrics", {}), "lpips_status": p["status"],
                              "display_frame": p["frame"], "policies": p["diagnostics"], "policy_errors": p["policy_errors"]} for p in prepared]}
    (out / "camera_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for p in prepared:
        scene_html(p, out)
    write_index(prepared, out, snapshot.get("captured_at", "unknown"))
    print(f"Open locally: {(out / 'index.html').resolve()}", flush=True)
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--snapshot", type=Path)
    inputs.add_argument("--work", type=Path)
    parser.add_argument("--case", default="local-scene")
    parser.add_argument("--lpips", type=float, help="Optional historical LPIPS for --work, display only")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.work:
        work, out = args.work.resolve(), args.out.resolve()
        if out == work or work in out.parents:
            parser.error("--out must be outside the prepared scene directory")
        snapshot = work_snapshot(work, args.case, args.lpips)
    else:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    generate(snapshot, args.out)


if __name__ == "__main__":
    main()
