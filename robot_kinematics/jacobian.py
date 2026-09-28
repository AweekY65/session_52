"""几何雅可比（Geometric Jacobian）。

``J(q)`` 是 6 x n 矩阵（n = 可动关节数），把关节速度映射为末端在
**base（世界）系** 下的空间速度 ``[v; omega]``：

- 上 3 行（平移部分）：``v = J_v @ q_dot``
- 下 3 行（旋转部分）：``omega = J_w @ q_dot``

列公式（``z_i`` 为关节轴世界系方向，``p_i`` 为关节原点世界系位置，
``p_e`` 为末端位置）：

- revolute / continuous：``[z_i x (p_e - p_i); z_i]``
- prismatic：             ``[z_i; 0]``

fixed joint 不产生列。解析雅可比与中心有限差分数值雅可比互相验证。
"""

from __future__ import annotations

import numpy as np

from .chain import KinematicChain
from .fk import forward_kinematics
from .transforms import matrix_to_rotvec


def jacobian(chain: KinematicChain, q) -> np.ndarray:
    """计算 base 系下 6 x dof 的几何雅可比。"""
    q = np.asarray(q, dtype=float).reshape(-1)
    link_poses, axes_world, origins_world = chain.accumulate_transforms(q)
    p_end = link_poses[-1][:3, 3]

    n = chain.dof
    j = np.zeros((6, n))
    for i, joint in enumerate(chain.movable_joints):
        z = axes_world[i]
        if joint.joint_type == "prismatic":
            j[:3, i] = z
        else:  # revolute / continuous
            j[:3, i] = np.cross(z, p_end - origins_world[i])
            j[3:, i] = z
    return j


def finite_difference_jacobian(
    chain: KinematicChain,
    q,
    delta: float = 1e-6,
) -> np.ndarray:
    """用中心有限差分近似 6 x dof 雅可比。

    - 平移行：末端位置对 q_i 的中心差分。
    - 旋转行：末端姿态对 q_i 的中心差分。姿态不在欧氏空间，
      因此用 SO(3) 对数映射 ``log(R(q+d) R(q-d)^T)`` 除以 ``2 delta``，
      得到的正是角速度块 ``J_w``。
    """
    q0 = np.asarray(q, dtype=float).reshape(-1)
    n = q0.size
    j_num = np.zeros((6, n))
    for i in range(n):
        step = delta * max(1.0, abs(q0[i]))
        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[i] += step
        q_minus[i] -= step
        t_plus = forward_kinematics(chain, q_plus)
        t_minus = forward_kinematics(chain, q_minus)
        j_num[:3, i] = (t_plus[:3, 3] - t_minus[:3, 3]) / (2.0 * step)
        drot = matrix_to_rotvec(t_plus[:3, :3] @ t_minus[:3, :3].T)
        j_num[3:, i] = drot / (2.0 * step)
    return j_num
