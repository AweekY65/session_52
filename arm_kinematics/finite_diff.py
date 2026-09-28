"""雅可比的中心有限差分验证。

位置部分::

    Jv[:, i] ≈ (p(q + e_i·d) - p(q - e_i·d)) / (2d)

旋转部分使用与 :mod:`arm_kinematics.transforms` 中相同的旋转向量约定。
令 ``w+ = log(R(q+) @ R(q)^T)``、``w- = log(R-) @ R(q)^T)``（末端相对
当前姿态的增量，在基座系下），则::

    Jw[:, i] ≈ (w+ - w-) / (2d)

若要验证链上某个中间 link 的雅可比，先用
``robot.extract_chain(end_link=link_name)`` 取子链，再调用本模块。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import coerce_q, forward_kinematics
from .transforms import rotation_to_axis_angle
from .urdf_loader import RobotModel


@dataclass
class JacobianCheck:
    """有限差分比对结果。"""

    jac_analytic: np.ndarray
    jac_numeric: np.ndarray
    abs_err: np.ndarray          # 逐元素绝对误差
    max_err: float
    pos_max_err: float
    rot_max_err: float
    passed: bool


def finite_difference_jacobian(robot: RobotModel, q, *, eps: float = 1e-6) -> np.ndarray:
    """用中心有限差分计算 6×nq 数值雅可比（对应 robot.end_link）。"""
    q = coerce_q(robot, q)
    n = q.size
    jac_num = np.zeros((6, n))

    r0 = forward_kinematics(robot, q).end_transform[:3, :3]

    for i in range(n):
        qp = q.copy()
        qm = q.copy()
        qp[i] += eps
        qm[i] -= eps
        tp = forward_kinematics(robot, qp).end_transform
        tm = forward_kinematics(robot, qm).end_transform

        # 位置中心差分
        jac_num[:3, i] = (tp[:3, 3] - tm[:3, 3]) / (2.0 * eps)

        # 姿态增量（末端相对当前姿态，在基座系下的旋转向量）
        wp = rotation_to_axis_angle(tp[:3, :3] @ r0.T)
        wm = rotation_to_axis_angle(tm[:3, :3] @ r0.T)
        jac_num[3:, i] = (wp - wm) / (2.0 * eps)

    return jac_num


def check_jacobian(robot: RobotModel, q, *, eps: float = 1e-6,
                   tol: float = 1e-5) -> JacobianCheck:
    """比较解析雅可比与有限差分数值雅可比。

    Args:
        tol: 逐元素误差阈值（位置单位为米/弧度，旋转单位为弧度）。
    """
    from .kinematics import jacobian

    q = coerce_q(robot, q)
    j_ana = jacobian(robot, q)
    j_num = finite_difference_jacobian(robot, q, eps=eps)
    err = np.abs(j_ana - j_num)
    pos_max = float(np.max(err[:3])) if err.size else 0.0
    rot_max = float(np.max(err[3:])) if err.size else 0.0
    max_err = float(np.max(err)) if err.size else 0.0
    return JacobianCheck(
        jac_analytic=j_ana,
        jac_numeric=j_num,
        abs_err=err,
        max_err=max_err,
        pos_max_err=pos_max,
        rot_max_err=rot_max,
        passed=max_err <= tol,
    )
