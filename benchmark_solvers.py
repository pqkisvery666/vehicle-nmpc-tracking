# -*- coding: utf-8 -*-
"""
求解器基准对比（benchmark_solvers.py）
=======================================

在**同一个** NMPC 问题（稳态跟踪 + 横向走廊 + 转向速率约束）上比较
CasADi 可用的几种 NLP 求解器，回答一个工程问题：

    "这个 MPC 问题用哪个求解器跑最快？实时余量还有多少？"

被比较的求解器
--------------
  ipopt     内点法，通用、稳健，本项目的默认选择
  fatrop    面向最优控制问题（OCP）的专用内点法，利用时间维稀疏结构
  sqpmethod CasADi 自带的 SQP（QP 子问题交给 qrqp）
  qrsqp      CasADi 自带的 SQP（在线活跃集 QP）

方法
----
每个求解器闭环跑 --steps 步（前 --warmup 步作为热机不计入统计），
记录单步求解耗时，输出 均值 / p95 / 最大 与实时余量
（控制周期 dt=0.1 s = 100 ms）。

输出：终端表格 + results/solver_benchmark.md
用法：python benchmark_solvers.py --steps 150 --warmup 20

注意：sqpmethod / qrsqp 会向控制台打印迭代日志（它们走自己的输出通道，
fd 重定向无法完全屏蔽）。本脚本已尝试 fd 级重定向，残留日志属正常现象，
不影响测量结果；最终表格与 Markdown 是干净的。
"""

import argparse
import contextlib
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

from model import rk4_step
from track import make_track
from mpc import NMPC
import simulate as sim

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

SOLVERS = ["ipopt", "fatrop", "sqpmethod", "qrsqp"]

# C 标准库句柄（用于刷新 C 层输出缓冲）：Windows 用 msvcrt，类 Unix 用主程序句柄
_libc = ctypes.CDLL("msvcrt" if sys.platform.startswith("win") else None)


@contextlib.contextmanager
def suppress_native_output():
    """
    屏蔽 C 层 stdout / stderr：部分求解器（sqpmethod / qrsqp）会直接 printf
    迭代日志。要点有两个：
      1. fd 重定向（sys.stdout 重定向拦不住 C 层输出）；
      2. 恢复 fd 之前必须 fflush C 标准库缓冲区 —— 否则日志只是被缓存，
         等缓冲区满或进程退出时才 flush 出来，仍然会刷屏。
    只影响当前进程，不涉及系统设置。
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            _libc.fflush(None)      # 关键：把 C 缓冲区内容冲进 devnull
        except Exception:
            pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def bench_one(solver, track, steps, warmup, corridor, rate, v_ref):
    """用指定求解器闭环跑一段，返回统计（不可用则返回 None + 原因）。"""
    try:
        mpc = NMPC(dt=sim.DT, N=sim.HORIZON, lane_half_width=corridor,
                   delta_rate_max=rate, solver=solver, name=solver)
    except Exception as exc:
        return None, f"构建失败：{str(exc).splitlines()[0][:80]}"

    x = np.array([track.P[0, 0], track.P[0, 1], track.psi[0], v_ref])
    i0, u_prev = 0, np.zeros(2)
    times, fails = [], 0

    for k in range(steps):
        refs = sim.make_refs(track, i0, mpc.N, v_ref)
        t0 = time.perf_counter()
        try:
            with suppress_native_output():
                res = mpc.solve(x, refs, u_prev=u_prev)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
        dt_ms = (time.perf_counter() - t0) * 1e3

        if not res["ok"]:
            fails += 1
            if fails > 5:
                return None, (f"持续求解失败（status={res['status']!r}，"
                              f"可能是该求解器不适用此问题）")
            continue                       # 保持上一拍控制继续跑

        if k >= warmup:
            times.append(dt_ms)

        u_prev = res["u0"]
        x = rk4_step(x, u_prev, sim.DT)
        i0 = track.nearest(x[:2], hint=i0)

    times = np.asarray(times)
    if times.size == 0:
        return None, "没有有效计时样本"

    return dict(
        solver=solver,
        n=int(times.size),
        mean=float(times.mean()),
        p50=float(np.percentile(times, 50)),
        p95=float(np.percentile(times, 95)),
        max=float(times.max()),
        margin=float(sim.DT * 1e3 / times.mean()),   # 单步耗时相对控制周期的余量倍数
        fails=fails,
    ), None


def main():
    ap = argparse.ArgumentParser(description="NMPC 求解器基准对比")
    ap.add_argument("--steps", type=int, default=150, help="每个求解器跑的步数")
    ap.add_argument("--warmup", type=int, default=20, help="热机步数（不计入统计）")
    ap.add_argument("--lane", type=float, default=sim.LANE_HALF_WIDTH,
                    help="走廊半宽 [m]")
    ap.add_argument("--delta-rate", type=float, default=sim.DELTA_RATE_MAX,
                    help="转向速率上限 [rad/s]")
    args = ap.parse_args()

    track = make_track(R0=32.0, ripple=0.18, harmonic=3, spacing=0.5)
    print(f"问题规模：N={sim.HORIZON}，dt={sim.DT}s，走廊 ±{args.lane:g} m，"
          f"转向速率 {args.delta_rate:g} rad/s")
    print(f"每个求解器 {args.steps} 步（前 {args.warmup} 步热机）...\n")

    rows, unavailable = [], []
    for solver in SOLVERS:
        print(f"  正在测试 {solver} ...", flush=True)
        stats, err = bench_one(solver, track, args.steps, args.warmup,
                               args.lane, args.delta_rate, sim.V_REF)
        if stats is None:
            unavailable.append((solver, err))
            print(f"    -> 不可用：{err}")
        else:
            rows.append(stats)
            print(f"    -> 均值 {stats['mean']:.1f} ms | "
                  f"p95 {stats['p95']:.1f} ms | 余量 {stats['margin']:.1f}×")

    # ---------------- 终端汇总 ----------------
    print("\n=================== 求解器基准（单步耗时） ===================")
    print(f"{'求解器':<12}{'样本':>6}{'均值ms':>10}{'p50':>9}{'p95':>9}"
          f"{'最大':>9}{'实时余量':>10}")
    for r in rows:
        print(f"{r['solver']:<12}{r['n']:>6}{r['mean']:>10.1f}{r['p50']:>9.1f}"
              f"{r['p95']:>9.1f}{r['max']:>9.1f}{r['margin']:>9.1f}×")
    for solver, err in unavailable:
        print(f"{solver:<12}{'—':>6}  不可用：{err}")

    # ---------------- 写出 Markdown ----------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# 求解器基准对比（NMPC 单步耗时）", "",
          f"- 问题：N={sim.HORIZON} 步预测时域，dt={sim.DT} s，"
          f"横向走廊 ±{args.lane:g} m，转向速率上限 {args.delta_rate:g} rad/s",
          f"- 协议：每个求解器闭环 {args.steps} 步，前 {args.warmup} 步热机不计入",
          "- 实时判据：单步耗时 < 控制周期 100 ms", "",
          "| 求解器 | 样本数 | 均值 [ms] | p50 [ms] | p95 [ms] | 最大 [ms] | 实时余量 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['solver']} | {r['n']} | {r['mean']:.1f} | {r['p50']:.1f} "
                  f"| {r['p95']:.1f} | {r['max']:.1f} | {r['margin']:.1f}× |")
    if unavailable:
        md += ["", "不可用的求解器："]
        md += [f"- `{s}`：{e}" for s, e in unavailable]
    (RESULTS_DIR / "solver_benchmark.md").write_text("\n".join(md),
                                                     encoding="utf-8")
    print(f"\nMarkdown 已写出 -> {RESULTS_DIR / 'solver_benchmark.md'}")


if __name__ == "__main__":
    main()
