# -*- coding: utf-8 -*-
"""
非线性模型预测控制（NMPC）—— 轨迹跟踪（v2 升级版）
==================================================

核心思想（滚动时域优化）：
    每个控制周期求解一个有限时域最优控制问题，只执行第一个控制量，
    下一时刻带着新的实测状态重新求解（预测 + 反馈）。

每个周期求解的问题（离散形式，N = 预测时域）：

    min_{s_0..s_N, u_0..u_{N-1}}  Σ_{k=0}^{N-1} [
            q_lat·e_lat(k)² + q_lon·e_lon(k)²            ← 路径坐标误差
          + q_psi·(1 - cos e_psi(k)) + q_v·e_v(k)²       ← 航向/速度误差
          + r_a·a_k² + r_d·δ_k²                          ← 控制代价
        ] + ρ·Σ s_k² + 终端代价（权重 ×2）

    s.t.  s_0 = s(t)                        当前实测状态（缺了它车会"从梦里出发"）
          s_{k+1} = f_RK4(s_k, u_k)         车辆动力学（RK4 离散）
          a ∈ [a_min, a_max], δ ∈ [δ_min, δ_max], v ∈ [v_min, v_max]
          |e_lat(k)| ≤ w/2 + s_k,  s_k ≥ 0  走廊约束（s_k 为松弛变量，可选）
          |δ_k - δ_{k-1}| ≤ δ̇_max·dt        转向速率约束（执行器速率限制）

v2 相对 v1 的三处升级（以及为什么）：

1. **代价函数从"笛卡尔位置误差"改为"路径坐标误差"**
   v1 直接惩罚 (px - px_ref)² + (py - py_ref)²，把"沿路径的进度误差"和
   "横向偏差"混在一起，权重无法分别调；当参考点以 6 m/s 前进而车只有
   3 m/s 时，优化器只能"满油门 + 猛打方向切弯追赶"，起步瞬态里出现
   25.8° 的异常转角峰值。v2 把误差投影到路径切向/法向：
       e_lon = cosψ_r·Δx + sinψ_r·Δy   （沿路径，权重小，仅保持进度）
       e_lat = -sinψ_r·Δx + cosψ_r·Δy  （横向，权重大，直接对应"压线"）
   这与生产级 MPC 的路径跟踪（Frenet / 路径坐标）写法一致。

2. **新增横向走廊硬约束**（lane_half_width）
   v1 唯一的"空间信息"在代价函数里（软惩罚），无法保证车辆不出界；
   v2 直接把 |e_lat| ≤ w/2 写成不等式约束 —— 这才是 MPC 相对 PID/LQR
   最本质的优势（显式处理硬约束）。约束里用的是**参考法向**，对状态是
   线性约束，数值上很友好。

3. **松弛变量（soft_lane）与转向速率约束（delta_rate_max）**
   - 硬走廊在"初始状态已在走廊外"时会导致问题不可行，工程做法是加
     松弛变量 s_k ≥ 0 并对它重罚（ρ = 1e4）：既能恢复又尽量不越界；
   - 真实转向执行器有速率上限，v1 只有幅值限制。速率约束写成相邻控制量
     之差的线性不等式，实现简单但显著提升真实性。

实现：CasADi Opti 建模 + IPOPT 求解（也可切换 fatrop / sqpmethod / qrsqp
用于求解器基准对比，见 benchmark_solvers.py）。
"""

import os
import sys

import numpy as np
import casadi as cs

from model import WHEELBASE, rk4_step


def _setup_plugin_search_path():
    """
    Windows 兼容修复（CasADi >= 3.7 常见坑）：
    CasADi 在 Windows 上通过普通 LoadLibrary 加载求解器插件（ipopt 等），
    DLL 依赖搜索只认 PATH，不认自身所在目录，会报
    "Plugin 'ipopt' is not found / WIN32 126"。
    这里把 casadi 包目录补进 PATH（仅当前进程、幂等），非 Windows 平台跳过。
    """
    if not sys.platform.startswith("win"):
        return
    casadi_dir = os.path.dirname(os.path.abspath(cs.__file__))
    cur_path = os.environ.get("PATH", "")
    if casadi_dir not in cur_path:
        os.environ["PATH"] = casadi_dir + os.pathsep + cur_path


_setup_plugin_search_path()

# ---------------- 默认权重 ----------------
# 量纲直觉：横向误差 1 m 远比纵向进度误差"危险"，所以 q_lat >> q_lon；
# 航向用 1-cos 周期惩罚（小误差时 ≈ 0.5·err²）；控制项抑制抖动。
DEFAULT_WEIGHTS = dict(
    lat=40.0,     # 横向误差（压线程度）
    lon=2.0,      # 纵向误差（保持进度，权重小）
    psi=18.0,     # 航向误差（1 - cos）
    v=8.0,        # 速度误差
    a=0.6,        # 加速度代价
    delta=20.0,   # 前轮转角代价（抑制抖动）
)
TERMINAL_SCALE = 2.0            # 终端代价放大倍数
SLACK_WEIGHT = 1.0e4            # 松弛变量罚权重（soft_lane 时生效）

# ---------------- 默认约束（实车限幅）----------------
DEFAULT_BOUNDS = dict(
    a=(-3.0, 3.0),        # 加速度限幅 [m/s^2]
    delta=(-0.5, 0.5),    # 前轮转角限幅 [rad] (~28.6 度)
    v=(0.05, 8.0),        # 速度范围（下限防止倒车/原地转）
)

# ---------------- 求解器配置 ----------------
# fatrop 是面向最优控制问题的专用求解器（利用 OCP 结构，通常更快）；
# sqpmethod / qrsqp 是 CasADi 自带的 SQP 实现，用于 benchmark_solvers.py 对比。
_SOLVER_SETUPS = {
    "ipopt": ("ipopt", {"print_time": False,
                        "ipopt": {"print_level": 0, "max_iter": 500}}),
    # 说明：sqpmethod / qrsqp 会向控制台打印迭代日志（走各自的输出通道，
    # fd 重定向拦不住）；不强行用打印开关关它，因为不同 CasADi 版本
    # 的选项名不一致（试过 print_iteration / print_iter，某些版本会报
    # unknown option 导致求解失败）。跑 benchmark 时日志较多属正常现象。
    "fatrop": ("fatrop", {"print_time": False}),
    "sqpmethod": ("sqpmethod", {"print_time": False, "qpsol": "qrqp"}),
    "qrsqp": ("qrsqp", {"print_time": False}),
}

# 不同求解器返回的状态字符串并不统一（ipopt 用 'Solve_Succeeded'，
# fatrop 返回 '0'），因此 solve() 不盲信状态字符串，而是**自行核验解**：
# 数值有效、满足限幅、且逐点满足离散动力学，才算求解成功。
_SUCCESS_STATUSES = {"Solve_Succeeded", "Solved_To_Acceptable_Level",
                     "Success", "0"}
_FEAS_TOL = 1e-4


class NMPC:
    """基于 CasADi Opti 的路径跟踪 NMPC（支持走廊/速率约束与软约束）。"""

    def __init__(self, dt=0.1, N=25, weights=None, bounds=None,
                 lane_half_width=None, delta_rate_max=None,
                 soft_lane=False, slack_weight=SLACK_WEIGHT,
                 solver="ipopt", name="mpc"):
        """
        dt             : 控制/预测周期 [s]
        N              : 预测时域步数
        weights        : 权重 dict（lat/lon/psi/v/a/delta），缺省用 DEFAULT_WEIGHTS
        bounds         : 限幅 dict；默认 None = 实车限幅。
                         对照实验传更宽范围（如 a=(-10,10), delta=(-1.4,1.4),
                         v=(0.05,15)）用于展示"若无约束会发生什么"。
        lane_half_width: 走廊半宽 [m]；None = 不施加横向约束（自由运行）
        delta_rate_max : 转向速率上限 [rad/s]；None = 不施加
        soft_lane      : True 时走廊为软约束（引入松弛变量并重罚）
        slack_weight   : 松弛变量罚权重
        solver         : "ipopt"（默认）/ "fatrop" / "sqpmethod" / "qrsqp"
        """
        self.dt = dt
        self.N = N
        self.name = name
        self.lane_half_width = lane_half_width
        self.delta_rate_max = delta_rate_max
        self.soft_lane = bool(soft_lane) and (lane_half_width is not None)

        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.weights = w
        if bounds is None:
            bounds = {k: tuple(v) for k, v in DEFAULT_BOUNDS.items()}
        self.bounds = bounds

        if solver not in _SOLVER_SETUPS:
            raise ValueError(f"未知求解器 {solver}，可选：{list(_SOLVER_SETUPS)}")
        self.solver_name = solver

        # ---------- 1. 决策变量 ----------
        opti = cs.Opti()
        X = opti.variable(4, N + 1)          # 预测状态轨迹 (4 x N+1)
        U = opti.variable(2, N)              # 控制序列 (2 x N)
        S = opti.variable(1, N + 1) if self.soft_lane else None   # 松弛变量

        # ---------- 2. 参数（每次求解只更新数值，不重建问题）----------
        p_x0 = opti.parameter(4, 1)          # 当前实测状态
        p_ref = opti.parameter(4, N + 1)     # 参考轨迹（每列一个参考路点）
        p_u_prev = opti.parameter(2, 1)      # 上一拍实际执行的控制量

        # ---------- 3. 代价函数（路径坐标误差）----------
        J = 0.0
        for k in range(N + 1):
            cos_r = cs.cos(p_ref[2, k])      # 参考航向：切向/法向投影
            sin_r = cs.sin(p_ref[2, k])
            dx = X[0, k] - p_ref[0, k]
            dy = X[1, k] - p_ref[1, k]
            e_lon = dx * cos_r + dy * sin_r                # 沿路径方向
            e_lat = -dx * sin_r + dy * cos_r               # 横向（左正右负）
            e_psi = X[2, k] - p_ref[2, k]
            e_v = X[3, k] - p_ref[3, k]

            scale = TERMINAL_SCALE if k == N else 1.0
            J += scale * (w["lat"] * e_lat ** 2 + w["lon"] * e_lon ** 2
                          + w["psi"] * (1.0 - cs.cos(e_psi))
                          + w["v"] * e_v ** 2)
            if k < N:
                J += w["a"] * U[0, k] ** 2 + w["delta"] * U[1, k] ** 2
            if self.soft_lane:
                J += slack_weight * S[0, k] ** 2

        # ---------- 4. 等式约束：初始状态 + 动力学 ----------
        opti.subject_to(X[:, 0] == p_x0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == rk4_step(X[:, k], U[:, k], dt))

        # ---------- 5. 不等式约束 ----------
        # 5.1 状态限幅（速度）
        v_lo, v_hi = bounds.get("v", DEFAULT_BOUNDS["v"])
        for k in range(N + 1):
            opti.subject_to(opti.bounded(v_lo, X[3, k], v_hi))

        # 5.2 控制限幅 + 数值安全（tan 在 ±π/2 奇异）
        a_lo, a_hi = bounds.get("a", DEFAULT_BOUNDS["a"])
        d_lo, d_hi = bounds.get("delta", DEFAULT_BOUNDS["delta"])
        for k in range(N):
            opti.subject_to(opti.bounded(a_lo, U[0, k], a_hi))
            opti.subject_to(opti.bounded(d_lo, U[1, k], d_hi))
            opti.subject_to(opti.bounded(-1.4, U[1, k], 1.4))

        # 5.3 横向走廊约束（硬约束 / 带松弛变量的软约束）
        if lane_half_width is not None:
            half = float(lane_half_width)
            for k in range(N + 1):
                cos_r = cs.cos(p_ref[2, k])
                sin_r = cs.sin(p_ref[2, k])
                e_lat = (-(X[0, k] - p_ref[0, k]) * sin_r
                         + (X[1, k] - p_ref[1, k]) * cos_r)
                if self.soft_lane:
                    opti.subject_to(S[0, k] >= 0)
                    opti.subject_to(e_lat <= half + S[0, k])
                    opti.subject_to(e_lat >= -half - S[0, k])
                else:
                    opti.subject_to(opti.bounded(-half, e_lat, half))

        # 5.4 转向速率约束（相邻控制量之差 + 与上一拍的跨周期衔接）
        if delta_rate_max is not None:
            dmax = float(delta_rate_max) * dt
            opti.subject_to(opti.bounded(-dmax, U[1, 0] - p_u_prev[1], dmax))
            for k in range(1, N):
                opti.subject_to(
                    opti.bounded(-dmax, U[1, k] - U[1, k - 1], dmax))

        # ---------- 6. 求解器 ----------
        opti.minimize(J)
        solver_name, solver_opts = _SOLVER_SETUPS[solver]
        opti.solver(solver_name, solver_opts)

        self.opti = opti
        self.X, self.U, self.S = X, U, S
        self.p_x0, self.p_ref, self.p_u_prev = p_x0, p_ref, p_u_prev
        self._last_sol = None      # 热启动缓存（失败时清空，避免坏初值传染）

    # ------------------------------------------------------------------
    def solve(self, x0, refs, u_prev=None):
        """
        求解一次 MPC。

        x0     : 当前实测状态 (4,)
        refs   : 参考轨迹，形状 (4, N+1)（时间在列）或 (N+1, 4)（时间在行）
        u_prev : 上一拍实际执行的控制量 (2,)；用于转向速率的跨周期衔接

        返回 dict: ok / status / u0 / X / U / slack_max
        """
        opti = self.opti
        N = self.N

        refs = np.asarray(refs, dtype=float)
        if refs.ndim == 2 and refs.shape[0] != 4 and refs.shape[1] == 4:
            refs = refs.T

        opti.set_value(self.p_x0, np.asarray(x0, dtype=float).reshape(4, 1))
        opti.set_value(self.p_ref, refs)
        opti.set_value(self.p_u_prev,
                       np.zeros(2) if u_prev is None
                       else np.asarray(u_prev, dtype=float).reshape(2, 1))

        # 热启动：上一拍最优解整体平移一位作为本拍初值（显著减少迭代次数）
        if self._last_sol is not None:
            Xg = np.column_stack([self._last_sol["X"][:, 1:],
                                  self._last_sol["X"][:, -1:]])
            Ug = np.column_stack([self._last_sol["U"][:, 1:],
                                  self._last_sol["U"][:, -1:]])
            opti.set_initial(self.X, Xg)
            opti.set_initial(self.U, Ug)
        else:
            Xg = np.array(refs, dtype=float, copy=True)
            Xg[:, 0] = np.asarray(x0, dtype=float)
            opti.set_initial(self.X, Xg)
            opti.set_initial(self.U, np.zeros((2, N)))
        if self.soft_lane:
            opti.set_initial(self.S, np.zeros((1, N + 1)))

        try:
            sol = opti.solve()
            status = opti.return_status()
        except Exception as exc:                    # 失败则清空热启动缓存
            self._last_sol = None
            return {"ok": False, "status": str(exc).splitlines()[0],
                    "u0": None, "slack_max": None}

        Xs = np.asarray(sol.value(self.X), dtype=float)
        Us = np.asarray(sol.value(self.U), dtype=float)
        slack_max = float(np.max(sol.value(self.S))) if self.soft_lane else 0.0

        ok = status in _SUCCESS_STATUSES and self._solution_ok(Xs, Us)
        if ok:
            self._last_sol = {"X": Xs, "U": Us}
        else:
            self._last_sol = None       # 解不可信 → 丢弃热启动缓存
        return {"ok": ok, "status": status, "u0": Us[:, 0].copy(),
                "X": Xs, "U": Us, "slack_max": slack_max}

    # ------------------------------------------------------------------
    def _solution_ok(self, Xs, Us, tol=_FEAS_TOL):
        """
        自行核验求解器返回的解（不盲信状态字符串）：

          1. 数值有效（无 NaN/Inf）；
          2. 满足幅值限幅（加速度 / 转角 / 速度，容差 tol）；
          3. 逐点满足离散动力学 s_{k+1} == f_RK4(s_k, u_k)（容差 tol）。

        这三条正好对应 docs/notes-mpc.md 的"验证清单"——把验证逻辑放到
        控制器内部，任何求解器（ipopt / fatrop / SQP 类）出问题都能被发现。
        """
        if not (np.all(np.isfinite(Xs)) and np.all(np.isfinite(Us))):
            return False

        a_lo, a_hi = self.bounds.get("a", DEFAULT_BOUNDS["a"])
        d_lo, d_hi = self.bounds.get("delta", DEFAULT_BOUNDS["delta"])
        v_lo, v_hi = self.bounds.get("v", DEFAULT_BOUNDS["v"])
        if Us[0].min() < a_lo - tol or Us[0].max() > a_hi + tol:
            return False
        if Us[1].min() < d_lo - tol or Us[1].max() > d_hi + tol:
            return False
        if Xs[3].min() < v_lo - tol or Xs[3].max() > v_hi + tol:
            return False

        for k in range(self.N):
            nxt = rk4_step(Xs[:, k], Us[:, k], self.dt)
            if float(np.abs(nxt - Xs[:, k + 1]).max()) > tol:
                return False
        return True
