# -*- coding: utf-8 -*-
"""
生成闭环跟踪动画 GIF（约束 NMPC · 完整 1 圈 · 无缝循环）
=========================================================

数据来源（优先级）：
  1. results/steady_data.npz —— simulate.py 跑稳态场景时缓存的数据（推荐，秒出）
  2. 若缓存不存在，则现场跑一遍稳态场景的约束组（约十几秒）

设计要点（与用户验收过的 v2 一致）：
  1. 只保留完整 1 圈 → 终点 ≈ 起点，GIF 循环时车不"瞬移"；
  2. 去掉坐标轴刻度/边框，只留赛道画面，构图干净；
  3. 车与车头朝向更大更清晰，轨迹线加粗；走廊边界用虚线画出。

用法: python animation.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from track import make_track
from mpc import NMPC
import simulate as sim

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

STRIDE = 2        # 每 2 个仿真步取 1 帧（0.2 s/帧）
FPS = 20
MARGIN = 6.0      # 画面外扩 [m]


def load_steady_run(track):
    """读取稳态场景的约束组数据；没有缓存就现场跑一遍。"""
    npz_path = RESULTS_DIR / "steady_data.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if "constrained_px" in data:
            print(f"读取缓存数据 -> {npz_path.name}")
            return (np.asarray(data["constrained_px"]),
                    np.asarray(data["constrained_py"]),
                    np.asarray(data["constrained_psi"]))

    print("未找到缓存数据，现场跑一遍稳态场景（约束组）...")
    cfg = sim.SCENARIOS["steady"]
    args = type("A", (), dict(horizon=sim.HORIZON,
                              delta_rate=sim.DELTA_RATE_MAX,
                              vref=sim.V_REF))()
    mpc = sim.build_controller("constrained", cfg, args)
    n_steps = int(np.ceil(1.05 * track.length / sim.V_REF / sim.DT))
    rec = sim.run_once(track, mpc, n_steps, "animation", v0=cfg["v0"],
                       wheelbase_real=cfg["wheelbase_real"], v_ref=sim.V_REF)
    return rec["px"], rec["py"], rec["psi"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    track = make_track(R0=32.0, ripple=0.18, harmonic=3, spacing=0.5)
    px, py, psi = load_steady_run(track)

    # ---- 裁剪到完整 1 圈：终点回到起点附近 → 循环无缝 ----
    arc = np.cumsum(np.hypot(np.diff(px), np.diff(py)))
    i_cut = int(np.searchsorted(arc, track.length))
    px, py, psi = px[: i_cut + 1], py[: i_cut + 1], psi[: i_cut + 1]
    close = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
    print(f"裁剪到 {px.size} 步 = 完整 1 圈 | 终点距起点 {close:.3f} m")

    # ---- 静态画面：赛道 + 走廊边界 ----
    left, right = track.lane_boundaries(2 * sim.LANE_HALF_WIDTH)
    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    ax.plot(track.P[:, 0], track.P[:, 1], color="0.55", lw=2.4)
    ax.plot(left[:, 0], left[:, 1], color="0.85", lw=1.0, ls="--")
    ax.plot(right[:, 0], right[:, 1], color="0.85", lw=1.0, ls="--")
    ax.plot(px[0], py[0], "*", color="k", ms=16, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(track.P[:, 0].min() - MARGIN, track.P[:, 0].max() + MARGIN)
    ax.set_ylim(track.P[:, 1].min() - MARGIN, track.P[:, 1].max() + MARGIN)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"约束 NMPC 闭环轨迹跟踪（完整 1 圈 · 目标 {sim.V_REF:g} m/s）",
                 fontsize=13, pad=10)

    (trail,) = ax.plot([], [], "-", color="#1f77b4", lw=2.2, alpha=0.9, zorder=4)
    (car,) = ax.plot([], [], "o", color="#d62728", ms=10, zorder=7)
    (nose,) = ax.plot([], [], "-", color="#d62728", lw=3.0, zorder=7)

    frame_idx = np.arange(0, px.size, STRIDE)
    print(f"共 {frame_idx.size} 帧动画，开始渲染 ...")

    def update(j):
        i = int(frame_idx[j])
        trail.set_data(px[: i + 1], py[: i + 1])
        car.set_data([px[i]], [py[i]])
        nose.set_data([px[i], px[i] + 1.6 * np.cos(psi[i])],
                      [py[i], py[i] + 1.6 * np.sin(psi[i])])
        return trail, car, nose

    anim = animation.FuncAnimation(fig, update, frames=frame_idx.size,
                                   interval=1000 // FPS, blit=False)
    out = RESULTS_DIR / "animation.gif"
    anim.save(str(out), writer=PillowWriter(fps=FPS), dpi=110)
    plt.close(fig)
    print(f"动画已保存 -> {out} ({frame_idx.size} 帧)")


if __name__ == "__main__":
    main()
