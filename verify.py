# -*- coding: utf-8 -*-
"""
自动验证脚本（verify.py）
==========================

把"MPC 实现是否正确、是否真的守约束"的检查固化成可执行脚本：
既是回归测试（改动后不怕改坏），也是仓库可信度证明（对应
docs/notes-mpc.md 的"验证清单"）。

检查项
------
  1. 初始状态约束  : 解出的 s0 必须等于传入的当前状态
  2. 动力学一致性  : s_{k+1} == f_RK4(s_k, u_k)，逐点比对整个预测时域
  3. 控制/状态限幅 : 加速度 / 转角 / 速度全程在约束内
  4. 横向走廊      : 标称场景全程不越界
  5. 转向速率      : 相邻控制量之差（含与上一拍的衔接）不超速率上限
  6. 软约束恢复    : 走廊外起步 → 松弛变量吸收越界 → 收敛回走廊内
  7. 硬约束不可行  : 走廊外起步 + 硬走廊 → 求解不可行（说明性实验：
                     这正是引入松弛变量的工程原因）
  8. 模型失配鲁棒  : 真实轴距 ≠ 模型轴距时，滚动反馈仍把误差压在小范围
  9. 求解成功率    : 标称运行 100% 成功
 10. 文档引用有效  : 所有 Markdown 里的图片/链接引用都必须真实存在
                     （防止改了输出文件名后 README 图片挂掉）

用法
----
    python verify.py            # 快速档（默认，约 700 次求解，几十秒）
    python verify.py --full     # 完整一圈（更慢，作为发布前检查）
退出码：0 = 全部通过；1 = 存在失败项。
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from model import rk4_step, WHEELBASE
from track import make_track
from mpc import NMPC, DEFAULT_BOUNDS
import simulate as sim

DT = sim.DT
TOL = 1e-6
PROJECT_DIR = Path(__file__).resolve().parent
_results = []


def find_broken_markdown_refs(project_dir=PROJECT_DIR):
    """
    扫描 Markdown 的图片/链接引用，返回指向不存在文件的引用列表。

    动机：README 里的图是相对路径引用，一旦输出文件名变化
    （例如结果图从 track_compare.png 改为 steady_track.png），
    引用就会静默失效、图挂掉。把这条检查放进自动验证，避免此类回归。

    扫描范围：本项目的 .md（递归）+ 上级目录（仓库根，如 MPC/）的 .md，
    但不进入 .venv / .idea 等环境目录。
    """
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
    root = Path(project_dir).resolve()
    skip = {".venv", ".idea", "__pycache__", ".git"}

    targets = [(md, root) for md in sorted(root.rglob("*.md"))
               if not any(part in skip for part in md.parts)]
    parent = root.parent
    targets += [(md, parent) for md in sorted(parent.glob("*.md"))]

    broken = []
    for md, base in targets:
        for ref in pattern.findall(md.read_text(encoding="utf-8")):
            if ref.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = (md.parent / ref.split("#")[0]).resolve()
            if not target.exists():
                try:
                    shown = md.relative_to(parent)
                except ValueError:
                    shown = md
                broken.append(f"{shown} -> {ref}")
    return broken


def check(name, passed, detail=""):
    _results.append((name, bool(passed), detail))
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}" + (f"  —  {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser(description="MPC 项目自动验证")
    ap.add_argument("--full", action="store_true",
                    help="用完整一圈的数据验证（默认 200 步快速档）")
    args = ap.parse_args()

    track = make_track(R0=32.0, ripple=0.18, harmonic=3, spacing=0.5)
    lap_steps = int(np.ceil(track.length / sim.V_REF / DT))
    n_steps = lap_steps if args.full else 200

    # 供 build_controller 使用的参数容器（与命令行默认一致）
    run_args = argparse.Namespace(horizon=sim.HORIZON,
                                  delta_rate=sim.DELTA_RATE_MAX,
                                  vref=sim.V_REF)
    cfg_steady = sim.SCENARIOS["steady"]
    cfg_recovery = sim.SCENARIOS["recovery"]
    cfg_mismatch = sim.SCENARIOS["mismatch"]

    # ==================================================================
    # 1 & 2. 单步解的不变式（初始状态 + 动力学一致性）
    # ==================================================================
    mpc = sim.build_controller("constrained", cfg_steady, run_args)
    t0 = track.tangent[0]
    x0 = np.array([track.P[0, 0], track.P[0, 1], track.psi[0], sim.V_REF])
    refs = sim.make_refs(track, 0, mpc.N, sim.V_REF)
    res = mpc.solve(x0, refs)

    check("1. 求解成功（标称初值）", res["ok"], f"status={res['status']}")
    if res["ok"]:
        err_s0 = float(np.abs(res["X"][:, 0] - x0).max())
        check("1. 初始状态约束 s0 == x0", err_s0 < 1e-8,
              f"最大误差 {err_s0:.2e}")

        dyn_err = 0.0
        for k in range(mpc.N):
            nxt = rk4_step(res["X"][:, k], res["U"][:, k], DT)
            dyn_err = max(dyn_err, float(np.abs(nxt - res["X"][:, k + 1]).max()))
        check("2. 动力学一致性（整个时域）", dyn_err < 1e-6,
              f"最大误差 {dyn_err:.2e}")

        # 优化器解本身是否满足限幅（比只看执行量更严格）
        a_lo, a_hi = DEFAULT_BOUNDS["a"]
        d_lo, d_hi = DEFAULT_BOUNDS["delta"]
        v_lo, v_hi = DEFAULT_BOUNDS["v"]
        ok_bounds = (res["U"][0].min() >= a_lo - TOL
                     and res["U"][0].max() <= a_hi + TOL
                     and res["U"][1].min() >= d_lo - TOL
                     and res["U"][1].max() <= d_hi + TOL
                     and res["X"][3].min() >= v_lo - TOL
                     and res["X"][3].max() <= v_hi + TOL)
        check("3. 优化解满足限幅（预测时域内）", ok_bounds,
              f"a∈[{res['U'][0].min():.3f},{res['U'][0].max():.3f}] "
              f"δ∈[{np.degrees(res['U'][1].min()):.2f},"
              f"{np.degrees(res['U'][1].max()):.2f}]°")
    else:
        check("2. 动力学一致性（整个时域）", False, "上一步求解失败，跳过")
        check("3. 优化解满足限幅（预测时域内）", False, "上一步求解失败，跳过")

    # ==================================================================
    # 3/4/5/9. 标称闭环：限幅、走廊、转向速率、成功率
    # ==================================================================
    rec = sim.run_once(track, mpc, n_steps, "verify-steady",
                       v0=cfg_steady["v0"],
                       wheelbase_real=cfg_steady["wheelbase_real"],
                       v_ref=sim.V_REF)
    a_lo, a_hi = DEFAULT_BOUNDS["a"]
    d_lo, d_hi = DEFAULT_BOUNDS["delta"]
    v_lo, v_hi = DEFAULT_BOUNDS["v"]

    ok_a = np.abs(rec["a"]).max() <= max(abs(a_lo), abs(a_hi)) + TOL
    ok_d = np.abs(rec["delta"]).max() <= max(abs(d_lo), abs(d_hi)) + TOL
    ok_v = (rec["v"].min() >= v_lo - 1e-6) and (rec["v"].max() <= v_hi + 1e-6)
    check(f"3. 闭环执行量满足限幅（{n_steps} 步）", ok_a and ok_d and ok_v,
          f"|a|max={np.abs(rec['a']).max():.3f}, "
          f"|δ|max={np.degrees(np.abs(rec['delta']).max()):.2f}°, "
          f"v∈[{rec['v'].min():.2f},{rec['v'].max():.2f}]")

    corridor = cfg_steady["corridor"]
    lat_max = np.abs(rec["lat_err"]).max()
    check(f"4. 横向走廊 ±{corridor:g} m 全程不越界", lat_max <= corridor + 1e-3,
          f"|e_lat|max={lat_max:.4f} m")

    if rec["delta"].size > 1:
        dmax_step = run_args.delta_rate * DT
        rate_err = np.abs(np.diff(rec["delta"])).max()
        first_err = abs(rec["delta"][0])          # u_prev 初值 = 0
        check("5. 转向速率约束", rate_err <= dmax_step + TOL
              and first_err <= dmax_step + TOL,
              f"|Δδ|max={rate_err:.4f} rad/步（上限 {dmax_step:.3f}），"
              f"首步 {first_err:.4f}")

    # ==================================================================
    # 6. 软约束恢复（走廊外起步）
    # ==================================================================
    mpc_soft = sim.build_controller("constrained", cfg_recovery, run_args)
    rec_rec = sim.run_once(track, mpc_soft, n_steps, "verify-recovery",
                           v0=cfg_recovery["v0"],
                           wheelbase_real=cfg_recovery["wheelbase_real"],
                           lateral_offset=cfg_recovery["lateral_offset"],
                           v_ref=sim.V_REF)
    tail = rec_rec["lat_err"][int(0.75 * rec_rec["lat_err"].size):]
    ok_recover = (rec_rec["slack"].max() > 0.5 and
                  np.abs(tail).max() <= cfg_recovery["corridor"] + 0.05)
    check("6. 软约束恢复（松弛变量吸收越界并回到走廊内）", ok_recover,
          f"slack峰值={rec_rec['slack'].max():.3f} m，"
          f"后 25% 的最大 |e_lat|={np.abs(tail).max():.4f} m "
          f"(走廊 ±{cfg_recovery['corridor']:g} m)")

    # ==================================================================
    # 7. 硬走廊不可行性（说明性实验）
    # ==================================================================
    mpc_hard = NMPC(dt=DT, N=sim.HORIZON, lane_half_width=cfg_recovery["corridor"],
                    delta_rate_max=run_args.delta_rate, soft_lane=False,
                    name="硬走廊-不可行性实验")
    t0 = track.tangent[0]
    n0 = np.array([-t0[1], t0[0]])
    pos0 = track.P[0] + cfg_recovery["lateral_offset"] * n0
    x_out = np.array([pos0[0], pos0[1], track.psi[0], sim.V_REF])
    res_hard = mpc_hard.solve(x_out, sim.make_refs(track, 0, mpc_hard.N, sim.V_REF))
    check("7. 硬走廊 + 走廊外初值 → 不可行（说明为何需要松弛变量）",
          not res_hard["ok"],
          f"求解状态：{(res_hard['status'] or '')[:60]}")

    # ==================================================================
    # 8. 模型失配鲁棒性（真实轴距 3.0 m，控制器按 2.5 m）
    # ==================================================================
    mpc_mm = sim.build_controller("constrained", cfg_mismatch, run_args)
    rec_mm = sim.run_once(track, mpc_mm, n_steps, "verify-mismatch",
                          v0=cfg_mismatch["v0"],
                          wheelbase_real=cfg_mismatch["wheelbase_real"],
                          v_ref=sim.V_REF)
    lat_mm = np.abs(rec_mm["lat_err"]).max()
    check("8. 模型失配下反馈仍把横向误差压在 0.10 m 内", lat_mm <= 0.10,
          f"实际轴距 {cfg_mismatch['wheelbase_real']:g} m vs "
          f"模型 {WHEELBASE:g} m，|e_lat|max={lat_mm:.4f} m")

    # ==================================================================
    # 9. 求解成功率
    # ==================================================================
    total_fail = (rec["fail_steps"] + rec_rec["fail_steps"]
                  + rec_mm["fail_steps"])
    total_steps = rec["ok_steps"] + rec_rec["ok_steps"] + rec_mm["ok_steps"]
    check("9. 标称运行 100% 求解成功", total_fail == 0,
          f"{total_steps} 步有效，{total_fail} 步失败")

    # ==================================================================
    # 10. 文档引用有效性（防止 README 图片/链接挂掉）
    # ==================================================================
    broken = find_broken_markdown_refs()
    check("10. Markdown 图片/链接引用均有效", not broken,
          "全部可解析" if not broken else
          f"失效引用 {len(broken)} 处：" + "; ".join(broken[:5]))

    # ==================================================================
    passed = sum(1 for _, ok, _ in _results if ok)
    print("\n=================== 验证汇总 ===================")
    for name, ok, _ in _results:
        print(f"  {'✔' if ok else '✘'} {name}")
    print(f"\n通过 {passed}/{len(_results)} 项")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
