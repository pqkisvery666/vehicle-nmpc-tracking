# -*- coding: utf-8 -*-
"""
车辆运动学模型（Kinematic Bicycle Model）
=========================================

为什么要用"自行车模型"而不是完整的车辆动力学模型？
    运动学模型忽略轮胎侧偏、质量转移等复杂因素，只描述"几何上的运动关系"，
    计算量小、参数少、在低速场景下精度足够。它是自动驾驶预测控制里最常用的
    起点模型，也是入门自动驾驶预测控制最经典的推导练习对象。

状态量（State）:  s = [px, py, psi, v]^T
    px, py : 后轴中心在全局坐标系下的位置 [m]
    psi    : 航向角（车头方向与 x 轴的夹角）[rad]
    v      : 纵向速度 [m/s]

控制量（Input）:  u = [a, delta]^T
    a     : 纵向加速度 [m/s^2]（油门/刹车）
    delta : 前轮转角 [rad]（方向盘转动的等效量）

推导（连续时间动力学）：
    后轴中心速度方向始终沿车身方向（无侧滑假设）：
        px_dot = v * cos(psi)
        py_dot = v * sin(psi)
    前轮转角 delta 使车身绕瞬时转动中心旋转：
        psi_dot = (v / L) * tan(delta)
    其中 L 为轴距（前轴到后轴距离），该式来自几何关系：转弯半径 R = L / tan(delta)，
    而角速度 = 线速度 / 半径。

    纵向：v_dot = a

写成向量形式：
    s_dot = f(s, u) = [v*cos(psi), v*sin(psi), (v/L)*tan(delta), a]^T

数值积分：
    仿真与 MPC 内部都用 4 阶 Runge-Kutta (RK4) 对连续模型做离散化，
    离散周期 dt 内精度远高于一阶欧拉，是实际工程常用做法。

实现说明：
    `rk4_step` 提供两条路径——纯 numpy 数值路径（闭环仿真推进真车，避免每步
    构造 CasADi 对象）与 CasADi 符号路径（MPC 内部构造等式约束），
    两者对同一个连续模型离散，保证"控制器模型"与"被控对象"一致。
"""

import numpy as np
import casadi as cs

# 轴距（前、后轴之间的距离），单位 m
WHEELBASE = 2.5


def cont_dynamics(s, u, L=WHEELBASE):
    """
    连续时间动力学 f(s, u)（CasADi 符号版本，供 MPC 构造约束）。

    参数同时兼容 numpy 数组与 casadi MX / DM。
    """
    v = s[3]
    dx = v * cs.cos(s[2])
    dy = v * cs.sin(s[2])
    dpsi = v / L * cs.tan(u[1])
    dv = u[0]
    return cs.vertcat(dx, dy, dpsi, dv)


def _cont_dynamics_np(s, u, L):
    """连续时间动力学 f(s, u)（纯 numpy 数值版本，供闭环仿真使用）。"""
    px_dot = s[3] * np.cos(s[2])
    py_dot = s[3] * np.sin(s[2])
    psi_dot = s[3] / L * np.tan(u[1])
    return np.array([px_dot, py_dot, psi_dot, u[0]], dtype=float)


def rk4_step(s, u, dt, L=WHEELBASE):
    """
    4 阶 Runge-Kutta 单步离散：s_{k+1} = rk4(s_k, u_k, dt)

    - 输入为 casadi 符号（MX/SX）→ 返回符号表达式（MPC 内构造动力学等式约束）
    - 输入为 numpy 数值 → 返回 numpy 数组（闭环仿真推进车辆状态）

    参数 L 可在仿真端单独指定，用于构造"控制器模型 ≠ 真实对象"的
    模型失配（model mismatch）实验。
    """
    if isinstance(s, (cs.MX, cs.SX)):
        k1 = cont_dynamics(s, u, L)
        k2 = cont_dynamics(s + dt / 2.0 * k1, u, L)
        k3 = cont_dynamics(s + dt / 2.0 * k2, u, L)
        k4 = cont_dynamics(s + dt * k3, u, L)
        return s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ---- 纯 numpy 快路径（无 CasADi 对象开销）----
    s = np.asarray(s, dtype=float)
    u = np.asarray(u, dtype=float)
    k1 = _cont_dynamics_np(s, u, L)
    k2 = _cont_dynamics_np(s + dt / 2.0 * k1, u, L)
    k3 = _cont_dynamics_np(s + dt / 2.0 * k2, u, L)
    k4 = _cont_dynamics_np(s + dt * k3, u, L)
    return s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
