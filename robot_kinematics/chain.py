"""从解析后的 URDF 模型构建单条机器人运动链并做结构校验。

校验内容
~~~~~~~~
- joint 引用的 link 不存在（缺失 link）
- 一个 link 有多个父 joint（多父节点 / 合流）
- 环结构（沿 child -> parent 上行时重复访问 link）
- 运动轴为零向量
- 模型不连通时要求显式指定 ``base_link``
- base 之下有多个叶子时要求显式指定 ``tip_link``

链内只保留 base -> tip 路径上的 link / joint；fixed joint 计入位姿传播，
但不计入自由度。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .errors import KinematicsError
from .transforms import make_transform, rotvec_to_matrix
from .urdf_parser import Joint, RobotModel


class KinematicChain:
    """base_link -> tip_link 的一条无分支运动链。

    Parameters
    ----------
    model:
        :func:`robot_kinematics.urdf_parser.parse_urdf` 的返回值。
    base_link / tip_link:
        末端 link 名称。模型不连通或有多个叶子时必须显式给出。
    """

    def __init__(
        self,
        model: RobotModel,
        base_link: Optional[str] = None,
        tip_link: Optional[str] = None,
    ) -> None:
        self.model = model

        # 所有 joint 引用的 link 必须存在（缺失 link 属于模型结构错误）。
        for joint in model.joints.values():
            for role, link_name in (("parent", joint.parent), ("child", joint.child)):
                if link_name not in model.links:
                    raise KinematicsError(
                        f"joint '{joint.name}' 的 {role} link '{link_name}' 不存在"
                    )

        # child link -> 唯一的父 joint（多父节点在此暴露）
        parent_of: dict[str, Joint] = {}
        for joint in model.joints.values():
            if joint.child in parent_of:
                raise KinematicsError(
                    f"link '{joint.child}' 同时被 joint "
                    f"'{parent_of[joint.child].name}' 和 '{joint.name}' 当作 child，"
                    f"存在多个父节点，无法构建运动链"
                )
            parent_of[joint.child] = joint

        base = self._select_base(model, parent_of, base_link)
        tip = self._select_tip(model, parent_of, base, tip_link)

        ordered_joints = self._walk_to_base(model, parent_of, base, tip)

        # 轴合法性：零向量轴没有定义的运动方向。
        for joint in ordered_joints:
            if np.linalg.norm(joint.axis) < 1e-12:
                raise KinematicsError(
                    f"joint '{joint.name}' 的 axis 是零向量，必须指定非零运动轴"
                )

        self.base_link: str = base
        self.tip_link: str = tip
        self.joints: list[Joint] = ordered_joints                      # base -> tip
        self.movable_joints: list[Joint] = [j for j in ordered_joints if not j.is_fixed]
        self.joint_names: list[str] = [j.name for j in self.movable_joints]

        # link 名顺序：base link + 每个 joint 的 child link
        link_names = [base]
        for joint in ordered_joints:
            link_names.append(joint.child)
        self.link_names: list[str] = link_names

    # ------------------------------------------------------------------ #
    # 构建辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_base(
        model: RobotModel, parent_of: dict[str, Joint], base_link: Optional[str]
    ) -> str:
        children = {joint.child for joint in model.joints.values()}
        roots = [name for name in model.links if name not in children]

        if base_link is not None:
            if base_link not in model.links:
                raise KinematicsError(f"base link '{base_link}' 不存在")
            return base_link

        if not roots:
            raise KinematicsError(
                "URDF 的 joint 图中找不到根 link（每个 link 都有父节点），存在环结构"
            )
        if len(roots) > 1:
            raise KinematicsError(
                f"URDF 中存在 {len(roots)} 个互不连通的根 link（不连通的树），"
                f"请显式指定 base_link；候选根：{sorted(roots)}"
            )
        return roots[0]

    @staticmethod
    def _select_tip(
        model: RobotModel,
        parent_of: dict[str, Joint],
        base: str,
        tip_link: Optional[str],
    ) -> str:
        # parent link -> 从它出发的 joints
        children_of: dict[str, list[Joint]] = {}
        for joint in model.joints.values():
            children_of.setdefault(joint.parent, []).append(joint)

        def leaf_under(start: str) -> Optional[str]:
            """从 start 沿唯一子链下行，遇到分叉返回 None，到叶子返回叶子名。"""
            current = start
            visited = {current}
            while True:
                outs = children_of.get(current, [])
                if not outs:
                    return current
                if len(outs) > 1:
                    return None
                nxt = outs[0].child
                if nxt in visited:
                    raise KinematicsError("URDF 的 joint 图存在环结构")
                visited.add(nxt)
                current = nxt

        if tip_link is not None:
            if tip_link not in model.links:
                raise KinematicsError(f"tip link '{tip_link}' 不存在")
            return tip_link

        leaf = leaf_under(base)
        if leaf is None:
            leaves = [name for name in model.links if not children_of.get(name)]
            raise KinematicsError(
                f"base link '{base}' 下存在多个末端 link，请显式指定 tip_link；"
                f"候选末端：{sorted(leaves)}"
            )
        return leaf

    @staticmethod
    def _walk_to_base(
        model: RobotModel,
        parent_of: dict[str, Joint],
        base: str,
        tip: str,
    ) -> list[Joint]:
        """从 tip 沿 parent 上行到 base，返回 base -> tip 的 joint 顺序。"""
        joints_reversed: list[Joint] = []
        current = tip
        visited = {current}
        while current != base:
            joint = parent_of.get(current)
            if joint is None:
                raise KinematicsError(
                    f"tip link '{tip}' 到 base link '{base}' 之间不连通："
                    f"link '{current}' 没有连接到父节点的 joint"
                )
            if joint.parent not in model.links:
                raise KinematicsError(
                    f"joint '{joint.name}' 引用的 parent link '{joint.parent}' 不存在"
                )
            joints_reversed.append(joint)
            current = joint.parent
            if current in visited:
                raise KinematicsError(
                    f"从 '{tip}' 上行时重复访问 link '{current}'，joint 图存在环结构"
                )
            visited.add(current)
        joints_reversed.reverse()
        return joints_reversed

    # ------------------------------------------------------------------ #
    # 基本属性
    # ------------------------------------------------------------------ #
    @property
    def dof(self) -> int:
        """可动关节数（fixed joint 不计入）。"""
        return len(self.movable_joints)

    @property
    def joint_limits(self) -> list[tuple[float, float]]:
        """每个可动关节的 (lower, upper)；continuous 为 (-inf, inf)。"""
        out: list[tuple[float, float]] = []
        for joint in self.movable_joints:
            if joint.limit is None:
                out.append((-np.inf, np.inf))
            else:
                out.append((joint.limit.lower, joint.limit.upper))
        return out

    # ------------------------------------------------------------------ #
    # 运动学原语
    # ------------------------------------------------------------------ #
    @staticmethod
    def joint_transform(joint: Joint, value: float) -> np.ndarray:
        """单个 joint 的局部齐次变换（origin + 关节运动）。"""
        t = joint.origin.matrix.copy()
        if joint.is_fixed:
            return t
        if joint.joint_type == "prismatic":
            # T = origin @ Translate(axis * q)：平移向量先经 origin 旋到父系。
            t[:3, 3] = t[:3, 3] + t[:3, :3] @ (value * joint.axis)
        else:  # revolute / continuous
            motion = np.eye(4)
            motion[:3, :3] = rotvec_to_matrix(value * joint.axis)
            t = t @ motion
        return t

    def validate_configuration(self, q) -> np.ndarray:
        """检查并返回可动关节向量：长度、有限性、关节限位。"""
        values = np.asarray(q, dtype=float).reshape(-1)
        if values.size != self.dof:
            raise KinematicsError(
                f"关节向量长度应为 {self.dof}（{self.dof} 个可动关节），实际为 {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise KinematicsError("关节值必须全部为有限数值")
        for value, joint in zip(values, self.movable_joints):
            if joint.limit is not None:
                lo, hi = joint.limit.lower, joint.limit.upper
                # 数值容差，允许恰好停在边界上的解。
                if value < lo - 1e-9 or value > hi + 1e-9:
                    raise KinematicsError(
                        f"关节 '{joint.name}' 的值 {value:.6g} 超出限位 [{lo}, {hi}]"
                    )
        return values

    def zero_configuration(self) -> np.ndarray:
        """零位关节向量（revolute 越限时夹到限位内，prismatic 同理）。"""
        q = np.zeros(self.dof)
        for i, (lo, hi) in enumerate(self.joint_limits):
            q[i] = min(hi, max(lo, 0.0))
        return q

    def accumulate_transforms(
        self, q, check_limits: bool = True
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """沿链累积变换，供 FK / Jacobian 共用。

        返回
        ~~~~
        link_poses:
            每个 link（含 base）相对 base 的 4x4 位姿，长度 n_links。
        joint_axes_world:
            每个可动关节轴在世界（base）系下的单位向量，长度 dof。
        joint_origins_world:
            每个可动关节原点在世界系下的位置，长度 dof。
        """
        if check_limits:
            values = self.validate_configuration(q)
        else:
            values = np.asarray(q, dtype=float).reshape(-1)
            if values.size != self.dof:
                raise KinematicsError(
                    f"关节向量长度应为 {self.dof}，实际为 {values.size}"
                )

        link_poses = [np.eye(4)]
        joint_axes_world: list[np.ndarray] = []
        joint_origins_world: list[np.ndarray] = []

        movable_index = 0
        current = np.eye(4)
        for joint in self.joints:
            value = 0.0
            if not joint.is_fixed:
                value = float(values[movable_index])
                movable_index += 1
            # joint frame = 应用 <origin> 后、关节运动前的框架；
            # URDF 的 axis 正是在该框架下表达的。
            joint_frame_world = current @ joint.origin.matrix
            t_local = self.joint_transform(joint, value)
            current = current @ t_local
            link_poses.append(current)
            if not joint.is_fixed:
                # URDF 几何约定：joint frame 的 z 轴（= axis）与父 link 系
                # 下的运动轴重合；joint frame 只是沿该轴放置、绕该轴定向的
                # 一点，<origin> 的平移通常沿运动轴方向。因此：
                #   - 轴线方向用 joint frame 的 R 转出来；
                #   - 轴线上的一点取 *父 link 原点*（在轴线上），
                #     不能取 joint origin（有沿轴偏移，不在转轴上）。
                axis_world = joint_frame_world[:3, :3] @ joint.axis
                origin_world = current[:3, 3].copy()
                joint_axes_world.append(axis_world)
                joint_origins_world.append(origin_world)
        return link_poses, joint_axes_world, joint_origins_world

    def clamp_to_limits(self, q) -> np.ndarray:
        """把关节向量逐维夹到限位内（continuous 不做处理）。"""
        q = np.asarray(q, dtype=float).reshape(-1)
        out = q.copy()
        for i, (lo, hi) in enumerate(self.joint_limits):
            if np.isfinite(lo):
                out[i] = max(lo, out[i])
            if np.isfinite(hi):
                out[i] = min(hi, out[i])
        return out
