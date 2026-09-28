"""正运动学 (FK) 与几何雅可比 (Jacobian)。

变换顺序（沿链逐个关节）::

    T_i(q_i) = T_origin_i * T_motion_i(q_i)

其中：
    - ``T_origin_i`` 来自 <origin xyz rpy>，是父 link 到关节运动框架的固定变换；
    - revolute / continuous: ``T_motion = Rot(axis, q)``
    - prismatic:             ``T_motion = Trans(axis * q)``
    - fixed:                 ``T_motion = I``

最终末端位姿 ``T_ee = T_1 T_2 ... T_n``，所有 link 位姿均表示在基座系下。

雅可比采用**基座系几何雅可比**（6×nq），关节轴与轴上点取关节运动前的
``T_origin`` 框架（即 ``T_origin@0`` 构型），与有限差分验证配套。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .urdf_loader import JointType, RobotModel
from .transforms import make_transform


@dataclass
class FKResult:
    """FK 计算结果。

    Attributes:
        link_transforms: 按基座到末端顺序排列的各 link 齐次变换（含 base 自身 I）
        link_map: link 名称 -> 4x4 变换
        joint_transforms: 关节名称 -> 运动后的关节框架（等于其 child link 框架）
        end_transform: 末端执行器（end_link）的 4x4 齐次变换
        q: 实际使用的广义坐标
    """

    link_transforms: list[np.ndarray]
    link_map: dict[str, np.ndarray]
    joint_transforms: dict[str, np.ndarray]
    end_transform: np.ndarray
    q: np.ndarray


def coerce_q(robot: RobotModel, q) -> np.ndarray:
    """把关节输入统一成与 ``robot.movable_joints`` 等长、同序的 ndarray。

    接受：
        - 序列：按 movable_joints 的顺序提供；
        - 字典：键为活动关节名（fixed 关节无需也无法给出）。
    """
    movable = robot.movable_joints
    if isinstance(q, dict):
        values = []
        for j in movable:
            if j.name not in q:
                raise KeyError(f"缺少活动关节 {j.name!r} 的关节值")
            values.append(float(q[j.name]))
        arr = np.array(values, dtype=float)
    else:
        arr = np.asarray(q, dtype=float).ravel()
        if arr.size != len(movable):
            raise ValueError(
                f"关节值数量 {arr.size} 与活动关节数量 {len(movable)} 不一致"
            )
    if not np.all(np.isfinite(arr)):
        raise ValueError("关节值必须全部为有限数")
    return arr


def check_q_within_limits(robot: RobotModel, q: np.ndarray) -> None:
    """检查 q 是否处于 revolute/prismatic 关节的 <limit> 内，越界抛出 ValueError。"""
    for value, joint in zip(q, robot.movable_joints):
        if joint.joint_type is JointType.CONTINUOUS:
            continue
        lo, hi = joint.lower, joint.upper
        if value < lo - 1e-12 or value > hi + 1e-12:
            raise ValueError(
                f"关节 {joint.name!r} 的值 {value:.6f} 超出限位 [{lo}, {hi}]"
            )


def _motion_transform(joint, value: float) -> np.ndarray:
    """关节运动部分 exp(axis * q)。"""
    t = np.eye(4)
    if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
        # Rodrigues 公式（theta 为标量，轴已归一化）
        axis = joint.axis
        c, s = np.cos(value), np.sin(value)
        x, y, z = axis
        t[:3, :3] = (
            c * np.eye(3)
            + (1 - c) * np.outer(axis, axis)
            + s * np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        )
    elif joint.joint_type is JointType.PRISMATIC:
        t[:3, 3] = joint.axis * value
    # fixed: 单位阵
    return t


def _origin_transform(joint) -> np.ndarray:
    return make_transform(joint.origin.xyz, joint.origin.rpy)


def chain_frames(robot: RobotModel, q: np.ndarray):
    """沿链计算每个关节的运动前框架 origin 与运动后框架。

    返回 (per_joint, link_map)：
        per_joint: [{"joint", "origin", "full"}, ...]，按 chain 顺序
        link_map:  link 名称 -> 基座位姿
    """
    link_map = {robot.base_link: np.eye(4)}
    per_joint = []
    t_cur = np.eye(4)  # base -> 当前（父）link
    q_index = 0

    for joint in robot.chain:
        t_origin_rel = _origin_transform(joint)
        t_origin = t_cur @ t_origin_rel  # base -> 关节运动框架（运动前）

        if joint.movable:
            value = float(q[q_index])
            q_index += 1
        else:
            value = 0.0
        t_full = t_origin @ _motion_transform(joint, value)

        per_joint.append({
            "joint": joint,
            "origin": t_origin,
            "full": t_full,
        })
        t_cur = t_full
        link_map[joint.child] = t_full

    return per_joint, link_map


def forward_kinematics(robot: RobotModel, q, *, check_limits: bool = False) -> FKResult:
    """正运动学。

    Args:
        robot: URDF 加载得到的模型
        q: 关节值（序列或 {关节名: 值} 字典，fixed 关节不占自由度）
        check_limits: 为 True 时检查 revolute/prismatic 关节是否越限

    Returns:
        FKResult，含各 link 与末端 end_link 的齐次变换。
    """
    q = coerce_q(robot, q)
    if check_limits:
        check_q_within_limits(robot, q)

    per_joint, link_map = chain_frames(robot, q)

    ordered_names = robot.link_names()
    link_transforms = [link_map[name] for name in ordered_names]
    joint_transforms = {item["joint"].name: item["full"] for item in per_joint}

    return FKResult(
        link_transforms=link_transforms,
        link_map=link_map,
        joint_transforms=joint_transforms,
        end_transform=link_map[robot.end_link],
        q=q,
    )


def jacobian(robot: RobotModel, q, *, link_name: str | None = None) -> np.ndarray:
    """计算基座系几何雅可比（6×nq）。

    行顺序为 ``[vx; vy; vz; wx; wy; wz]``，列顺序与 ``robot.movable_joints`` 一致。

    Args:
        link_name: 雅可比对应的末端 link；缺省为 robot.end_link。
                   必须是运动链上的 link（中间 link 时只使用其上游关节）。
    """
    q = coerce_q(robot, q)
    target = link_name or robot.end_link
    chain_names = robot.link_names()
    if target not in chain_names:
        raise ValueError(
            f"link {target!r} 不在 base({robot.base_link!r}) 到 end({robot.end_link!r}) 的链上"
        )
    last_index = chain_names.index(target)  # 使用 chain[:last_index] 个关节

    per_joint, _ = chain_frames(robot, q)
    p_ee = per_joint[last_index - 1]["full"][:3, 3] if last_index > 0 \
        else np.zeros(3)

    jac = np.zeros((6, robot.nq))
    col = 0
    for i, item in enumerate(per_joint[:last_index]):
        joint = item["joint"]
        if not joint.movable:
            continue
        t_origin = item["origin"]
        z_base = t_origin[:3, :3] @ joint.axis
        if joint.joint_type is JointType.PRISMATIC:
            jac[:3, col] = z_base
        else:  # revolute / continuous
            p_joint = t_origin[:3, 3]
            jac[:3, col] = np.cross(z_base, p_ee - p_joint)
            jac[3:, col] = z_base
        col += 1
    return jac
