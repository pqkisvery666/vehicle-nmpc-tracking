# -*- coding: utf-8 -*-
"""
主程序：闭环赛道上的 NMPC 轨迹跟踪仿真（v2）
==============================================

一次运行跑完 4 个场景，每个场景对比两种控制器配置：

  场景              | 目的
  ------------------|----------------------------------------------------
  steady   稳态巡航 | 从目标速度起步，考察弯道跟踪质量与约束遵守
  startup  起步瞬态 | 3 m/s 起步加速到巡航，考察阶跃响应（控制量是否合理）
  recovery 偏移恢复 | 初始横向偏离 2.5 m（在走廊外），考察软约束 + 恢复能力
  mismatch 模型失配 | 真实轴距 3.0 m 而控制器仍按 2.5 m 预测，考察反馈鲁棒性

  控制器           | 配置
  -----------------|------------------------------------------------------
  constrained 约束组 | 实车限幅 + 横向走廊（+转向速率约束）
  relaxed     对照   | 宽松限幅、无走廊、无速率约束 —— "若无约束会怎样"

输出（results/ 目录）：
  <scenario>_track.png        轨迹对比（含走廊边界与越界点标注）
  <scenario>_timeseries.png   速度/加速度/转角/横向误差（+松弛变量）时序对比
  <scenario>_data.npz         原始数据（供动画与画图复用，避免重复求解）
  run_config.json             本次运行的完整配置（可复现性）

用法示例：
    python simulate.py                     # 跑全部场景
    python simulate.py --scenario steady   # 只跑稳态场景
    python simulate.py --vref 7.0 --lane 2.5 --delta-rate 0.8
    python simulate.py --quick             # 快速模式（0.4 圈，迭代调参用）
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # 无界面后端，只存图（命令行/服务器友好）
import matplotlib.pyplot as plt

# 中文字体（Windows 用微软雅黑/黑体；其他平台回退默认字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False   # 正常显示负号

from model import rk4_step, WHEELBASE
from track import make_track
from mpc import NMPC, DEFAULT_BOUNDS, DEFAULT_WEIGHTS

# ---------------- 路径与默认配置 ----------------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

DT = 0.1              # 控制周期 [s]
V_REF = 6.0           # 目标巡航速度 [m/s]
HORIZON = 25          # 预测时域步数
LANE_HALF_WIDTH = 3.0     # 走廊半宽 [m]（约束组）
DELTA_RATE_MAX = 0.6      # 转向速率上限 [rad/s]（约束组）
X_OFFSET_RECOVERY = 2.5   # 偏移恢复场景的初始横向偏移 [m]

# 对照组的"宽松限幅"：展示无约束时会给出什么指令
RELAXED_BOUNDS = dict(a=(-10.0, 10.0), delta=(-1.4, 1.4), v=(0.05, 15.0))

SCENARIOS = {
    "steady": dict(
        title="稳态巡航",
        desc="从目标速度起步：考察弯道跟踪质量与约束遵守",
        v0=V_REF, lateral_offset=0.0, wheelbase_real=WHEELBASE,
        corridor=LANE_HALF_WIDTH, soft=False, relaxed_corridor=None),
    "startup": dict(
        title="起步瞬态",
        desc="3 m/s 起步加速到巡航：考察阶跃响应与控制量合理性",
        v0=3.0, lateral_offset=0.0, wheelbase_real=WHEELBASE,
        corridor=LANE_HALF_WIDTH, soft=False, relaxed_corridor=None),
    "recovery": dict(
        title="偏移恢复",
        desc="初始横向偏离 2.5 m（走廊外）：考察软约束与恢复能力",
        v0=V_REF, lateral_offset=X_OFFSET_RECOVERY, wheelbase_real=WHEELBASE,
        corridor=1.5, soft=True, relaxed_corridor=None),
    "mismatch": dict(
        title="模型失配",
        desc="真实轴距 3.0 m、控制器按 2.5 m 预测：考察滚动反馈的鲁棒性",
        v0=V_REF, lateral_offset=0.0, wheelbase_real=3.0,
        corridor=1.5, soft=False, relaxed_corridor=None),
}

COLORS = {"constrained": "#1f77b4", "relaxed": "#d62728"}
LABELS = {"constrained": "NMPC（限幅+走廊+速率约束）",
          "relaxed": "NMPC（宽松对照：无走廊/无速率约束）"}


# ======================================================================
# 参考轨迹
# ======================================================================
def make_refs(track, i0, N, v_ref):
    """
    以当前最近路点 i0 为起点，沿赛道取未来 N+1 步的参考路点。

    参考点按"目标速度的时间参数化"推进：每步前进 adv = v_ref·dt/spacing
    个路点（通常非整数，例 6 m/s、间距 0.5 m → 1.2），因此对相邻路点
    做线性插值——否则参考移动速度会被路点间距锁死（v1 曾踩此坑）。
    """
    adv = v_ref * DT / track.spacing
    ki = (i0 + adv * np.arange(N + 1)) % track.n
    i_lo = np.floor(ki).astype(int)
    frac = ki - i_lo
    i_hi = (i_lo + 1) % track.n
    P_lo, P_hi = track.P[i_lo], track.P[i_hi]

    return np.column_stack([
        P_lo[:, 0] * (1 - frac) + P_hi[:, 0] * frac,
        P_lo[:, 1] * (1 - frac) + P_hi[:, 1] * frac,
        track.psi[i_lo] * (1 - frac) + track.psi[i_hi] * frac,
        np.full(N + 1, v_ref),
    ])


# ======================================================================
# 控制器工厂
# ======================================================================
def build_controller(kind, cfg, args):
    """kind: 'constrained' / 'relaxed'"""
    common = dict(dt=DT, N=args.horizon)
    if kind == "constrained":
        return NMPC(bounds=None,
                    lane_half_width=cfg["corridor"],
                    delta_rate_max=args.delta_rate,
                    soft_lane=cfg["soft"],
                    name=f"{cfg['title']}-约束组", **common)
    return NMPC(bounds=RELAXED_BOUNDS, lane_half_width=None,
                delta_rate_max=None, soft_lane=False,
                name=f"{cfg['title']}-对照", **common)


# ======================================================================
# 闭环仿真
# ======================================================================
def run_once(track, mpc, n_steps, tag, v0, wheelbase_real,
             lateral_offset=0.0, v_ref=V_REF):
    """
    闭环跑一条赛道（直到步数用完或状态发散）。

    与 v1 的区别：
      - 初始状态可带横向偏移（构造"走廊外恢复"场景）；
      - "真实车辆"可用不同轴距积分（构造模型失配场景）；
      - 单步求解失败不再终止整个仿真，而是保持上一拍控制量并计数；
      - 最近路点查询带进度提示（局部窗口搜索，跑出窗口自动回退全量）。
    """
    rec = {k: [] for k in ("t", "px", "py", "psi", "v", "a", "delta",
                           "dev", "lat_err", "slack", "solve_ms")}

    # ---- 初始状态：赛道起点，可带横向偏移 ----
    i0 = 0
    t0 = track.tangent[i0]
    n0 = np.array([-t0[1], t0[0]])
    pos0 = track.P[i0] + lateral_offset * n0
    x_cur = np.array([pos0[0], pos0[1], track.psi[i0], v0])
    u_prev = np.zeros(2)

    ok_steps, fail_steps, consecutive_fail = 0, 0, 0

    for k in range(n_steps):
        refs = make_refs(track, i0, mpc.N, v_ref)
        t_start = time.perf_counter()
        res = mpc.solve(x_cur, refs, u_prev=u_prev)
        dt_solve = (time.perf_counter() - t_start) * 1e3

        if res["ok"]:
            u_applied = res["u0"]
            consecutive_fail = 0
        else:
            # 兜底：求解失败时保持上一拍控制量（工程上可替换为紧急制动）
            u_applied = u_prev.copy()
            fail_steps += 1
            consecutive_fail += 1
            if consecutive_fail <= 3 or consecutive_fail % 50 == 0:
                print(f"[{tag}] step {k}: 求解失败（保持上一拍控制）"
                      f" status={res['status'][:70]}")
            if consecutive_fail >= 30:
                print(f"[{tag}] 连续 {consecutive_fail} 步求解失败，终止")
                break

        # ---- 记录当前时刻 ----
        e_lat, _ = track.lateral_error(x_cur[:2], hint=i0)
        rec["t"].append(k * DT)
        rec["px"].append(x_cur[0]); rec["py"].append(x_cur[1])
        rec["psi"].append(x_cur[2]); rec["v"].append(x_cur[3])
        rec["a"].append(u_applied[0]); rec["delta"].append(u_applied[1])
        rec["lat_err"].append(e_lat)
        rec["dev"].append(abs(e_lat))
        rec["slack"].append(res.get("slack_max") or 0.0)
        rec["solve_ms"].append(dt_solve)

        # ---- 推进"真实车辆"（可用不同轴距模拟模型失配）----
        x_cur = rk4_step(x_cur, u_applied, DT, L=wheelbase_real)
        if not np.all(np.isfinite(x_cur)):
            print(f"[{tag}] step {k}: 状态发散，提前终止")
            break

        u_prev = u_applied
        i0 = track.nearest(x_cur[:2], hint=i0)
        ok_steps += 1

    for key in rec:
        rec[key] = np.asarray(rec[key])
    rec["ok_steps"] = ok_steps
    rec["fail_steps"] = fail_steps
    return rec


# ======================================================================
# 统计与打印
# ======================================================================
def summarize(name, rec, corridor=None):
    """打印统计指标；对空记录给出可读提示（v1 在此处会 IndexError 崩溃）。"""
    print(f"\n----- {name} -----")
    if rec["ok_steps"] == 0:
        print("  没有有效步数（首步即失败），请检查约束是否可行/初值是否合理")
        return dict(ok_steps=0)

    a, d = rec["a"], rec["delta"]
    lat = rec["lat_err"]
    stats = dict(
        ok_steps=rec["ok_steps"],
        fail_steps=rec["fail_steps"],
        duration=float(rec["t"][-1]),
        lat_abs_max=float(np.abs(lat).max()),
        lat_abs_mean=float(np.abs(lat).mean()),
        a_abs_max=float(np.abs(a).max()),
        delta_deg_max=float(np.degrees(np.abs(d).max())),
        v_min=float(rec["v"].min()), v_max=float(rec["v"].max()),
        solve_ms_mean=float(rec["solve_ms"].mean()),
        solve_ms_max=float(rec["solve_ms"].max()),
        rate_rad_max=float(np.abs(np.diff(d)).max()) if d.size > 1 else 0.0,
        slack_max=float(rec["slack"].max()),
    )
    if corridor is not None:
        viol = np.abs(lat) > corridor + 1e-9
        stats["viol_steps"] = int(viol.sum())
        stats["viol_max"] = float(max(0.0, (np.abs(lat) - corridor).max()))
        stats["margin_min"] = float(corridor - np.abs(lat).max())

    print(f"  有效步数 / 失败步数 : {stats['ok_steps']} / {stats['fail_steps']}")
    print(f"  行驶时间            : {stats['duration']:.1f} s")
    print(f"  横向误差 |e_lat|    : 均值 {stats['lat_abs_mean']:.4f} m，"
          f"峰值 {stats['lat_abs_max']:.4f} m")
    if corridor is not None:
        if stats["viol_steps"] == 0:
            print(f"  走廊(±{corridor:.2f} m)    : 全程未越界"
                  f"（最小裕量 {stats['margin_min']:.4f} m）")
        else:
            print(f"  走廊(±{corridor:.2f} m)    : 越界 {stats['viol_steps']} 步，"
                  f"最大越界 +{stats['viol_max']:.4f} m")
    print(f"  最大加速度 |a|      : {stats['a_abs_max']:.3f} m/s^2")
    print(f"  最大转角 |δ|        : {stats['delta_deg_max']:.2f} deg")
    print(f"  最大转向速率 |Δδ|   : {stats['rate_rad_max']:.4f} rad/步")
    print(f"  速度范围            : [{stats['v_min']:.2f}, {stats['v_max']:.2f}] m/s")
    print(f"  单步求解耗时        : 均值 {stats['solve_ms_mean']:.1f} ms，"
          f"最大 {stats['solve_ms_max']:.1f} ms")
    if rec["slack"].max() > 0:
        print(f"  松弛变量峰值        : {stats['slack_max']:.4f} m（软约束启用）")
    return stats


# ======================================================================
# 出图
# ======================================================================
def plot_tracks(track, recs, cfg, out_path, corridor):
    """轨迹对比图：中心线 + 走廊边界 + 两条轨迹（越界点单独标注）。"""
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(track.P[:, 0], track.P[:, 1], color="0.55", lw=2.0,
            label="赛道中心线")
    if corridor is not None:
        left, right = track.lane_boundaries(2 * corridor)
        ax.plot(left[:, 0], left[:, 1], color="0.75", lw=1.2, ls="--",
                label=f"走廊边界 ±{corridor:g} m")
        ax.plot(right[:, 0], right[:, 1], color="0.75", lw=1.2, ls="--")

    for kind, rec in recs.items():
        if rec["ok_steps"] == 0:
            continue
        ax.plot(rec["px"], rec["py"], color=COLORS[kind], lw=1.8,
                label=LABELS[kind])
        ax.plot(rec["px"][0], rec["py"][0], "o", color=COLORS[kind],
                ms=8, zorder=5)
        if corridor is not None:
            over = np.abs(rec["lat_err"]) > corridor + 1e-9
            if over.any():
                ax.plot(rec["px"][over], rec["py"][over], "x",
                        color=COLORS[kind], ms=5, mew=1.2,
                        label=f"{kind} 越界点（{int(over.sum())} 步）")

    ax.set_aspect("equal")
    ax.set_title(f"{cfg['title']}：{cfg['desc']}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  轨迹图 -> {out_path.name}")


def plot_time_series(recs, cfg, out_path, corridor, delta_rate_max, vref):
    """时序对比图：速度 / 加速度 / 转角 / 横向误差（软约束场景加松弛变量）。"""
    has_slack = any(rec["slack"].max() > 0 for rec in recs.values()
                    if rec["ok_steps"] > 0)
    n_rows = 5 if has_slack else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.2 * n_rows + 1),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for kind, rec in recs.items():
        if rec["ok_steps"] == 0:
            continue

        # 1) 速度
        ax = axes[0]
        ax.plot(rec["t"], rec["v"], color=COLORS[kind], lw=1.5, label=LABELS[kind])
        # 2) 加速度
        ax = axes[1]
        ax.plot(rec["t"], rec["a"], color=COLORS[kind], lw=1.5)
        # 3) 转角
        ax = axes[2]
        ax.plot(rec["t"], np.degrees(rec["delta"]), color=COLORS[kind], lw=1.5)
        # 4) 横向误差（带符号）
        ax = axes[3]
        ax.plot(rec["t"], rec["lat_err"], color=COLORS[kind], lw=1.5)
        # 5) 松弛变量
        if has_slack:
            ax = axes[4]
            ax.plot(rec["t"], rec["slack"], color=COLORS[kind], lw=1.5,
                    label=LABELS[kind])

    # ---- 参考线与限幅带（约束组的限幅；对照组未施加，图注中说明）----
    a_lo, a_hi = DEFAULT_BOUNDS["a"]
    d_lo, d_hi = np.degrees(DEFAULT_BOUNDS["delta"])
    v_lo, v_hi = DEFAULT_BOUNDS["v"]

    ax = axes[0]
    ax.axhline(vref, color="k", ls="--", lw=1, label=f"目标速度 {vref:g} m/s")
    ax.axhline(v_hi, color="0.45", ls=":", lw=1,
               label=f"约束组限速上限 {v_hi:g} m/s")
    ax.axhline(v_lo, color="0.45", ls=":", lw=1)
    ax.set_ylabel("v [m/s]"); ax.legend(fontsize=8, ncol=3, loc="lower right")

    ax = axes[1]
    ax.axhspan(a_lo, a_hi, color="0.88", zorder=0,
               label=f"约束组限幅 [{a_lo:g}, {a_hi:g}] m/s²")
    ax.set_ylabel("a [m/s²]"); ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    ax.axhspan(d_lo, d_hi, color="0.88", zorder=0,
               label=f"约束组限幅 ±{d_hi:.1f}°（对照组未施加）")
    if delta_rate_max is not None:
        dmax_deg = np.degrees(delta_rate_max * DT)
        ax.set_title(f"转向速率约束：|Δδ| ≤ {dmax_deg:.2f}°/步"
                     f"（= {delta_rate_max:g} rad/s × dt）",
                     fontsize=9, loc="right")
    ax.set_ylabel("δ [deg]"); ax.legend(fontsize=8, loc="lower right")

    ax = axes[3]
    if corridor is not None:
        ax.axhspan(-corridor, corridor, color="0.88", zorder=0,
                   label=f"走廊 ±{corridor:g} m（仅约束组施加）")
    ax.set_ylabel("e_lat [m]"); ax.legend(fontsize=8, loc="lower right")

    if has_slack:
        ax = axes[4]
        ax.set_ylabel("slack [m]")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("时间 [s]")
    fig.suptitle(f"{cfg['title']}：{cfg['desc']}", y=0.997, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  时序图 -> {out_path.name}")


# ======================================================================
# 单场景执行
# ======================================================================
def run_scenario(name, cfg, args):
    """跑一个场景的两种控制器 → 出图 + 存数据 + 返回统计。"""
    print(f"\n==================== [{name}] {cfg['title']} ====================")
    print(f"  {cfg['desc']}")

    track = make_track(R0=32.0, ripple=0.18, harmonic=3, spacing=0.5)
    lap_steps = track.length / args.vref / DT
    n_steps = int(np.ceil((0.4 if args.quick else 1.05) * lap_steps))
    print(f"  赛道 {track.length:.0f} m，本轮 {n_steps} 步"
          f"（{'快速模式' if args.quick else '约 1 圈'}）")

    recs, stats = {}, {}
    for kind in ("constrained", "relaxed"):
        mpc = build_controller(kind, cfg, args)
        rec = run_once(track, mpc, n_steps, f"{cfg['title']}-{kind}",
                       v0=cfg["v0"], wheelbase_real=cfg["wheelbase_real"],
                       lateral_offset=cfg["lateral_offset"], v_ref=args.vref)
        recs[kind] = rec
        stats[kind] = summarize(mpc.name, rec, corridor=cfg["corridor"])

    # ---- 出图 ----
    plot_tracks(track, recs, cfg, RESULTS_DIR / f"{name}_track.png",
                cfg["corridor"])
    plot_time_series(recs, cfg, RESULTS_DIR / f"{name}_timeseries.png",
                     cfg["corridor"], args.delta_rate, args.vref)

    # ---- 数据缓存（供动画/画图复用，不必重复求解）----
    np.savez(RESULTS_DIR / f"{name}_data.npz",
             **{f"{kind}_{key}": val for kind, rec in recs.items()
                for key, val in rec.items()})
    return stats


# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="车辆 NMPC 轨迹跟踪闭环仿真")
    p.add_argument("--scenario", default="all",
                   choices=["all"] + list(SCENARIOS),
                   help="要运行的场景（默认 all）")
    p.add_argument("--vref", type=float, default=V_REF, help="目标巡航速度 [m/s]")
    p.add_argument("--horizon", type=int, default=HORIZON, help="预测时域步数")
    p.add_argument("--delta-rate", type=float, default=DELTA_RATE_MAX,
                   help="转向速率上限 [rad/s]")
    p.add_argument("--quick", action="store_true",
                   help="快速模式：只跑 0.4 圈（调参用，结果不作为最终数据）")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    all_stats = {}
    for name in names:
        all_stats[name] = run_scenario(name, SCENARIOS[name], args)

    # ---- 配置存档（复现性）----
    config = dict(dt=DT, v_ref=args.vref, horizon=args.horizon,
                  lane_half_width=LANE_HALF_WIDTH,
                  delta_rate_max=args.delta_rate,
                  weights=DEFAULT_WEIGHTS, bounds=DEFAULT_BOUNDS,
                  relaxed_bounds=RELAXED_BOUNDS,
                  wheelbase_model=WHEELBASE, quick=args.quick,
                  scenarios={k: {kk: vv for kk, vv in v.items()}
                             for k, v in SCENARIOS.items() if k in names})
    with open(RESULTS_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # ---- 汇总表 ----
    print("\n\n===================== 汇总 =====================")
    header = (f"{'场景':<10}{'控制器':<12}{'e_lat峰值':>10}"
              f"{'越界步数':>10}{'|a|峰值':>10}{'|δ|峰值':>10}"
              f"{'耗时均值':>10}")
    print(header)
    for name in names:
        for kind, st in all_stats[name].items():
            if st.get("ok_steps", 0) == 0:
                continue
            print(f"{name:<10}{kind:<12}{st['lat_abs_max']:>10.4f}"
                  f"{st.get('viol_steps', 0):>10d}{st['a_abs_max']:>10.3f}"
                  f"{st['delta_deg_max']:>10.2f}{st['solve_ms_mean']:>10.1f}")
    print(f"\n结果已保存到 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
