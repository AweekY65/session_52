"""轻量 URDF 解析器（标准库 xml.etree，无 ROS 依赖）。

解析内容：``<link>``、``<joint>`` 的 name/type、``<origin>`` 的 xyz/rpy、
``<axis>`` 以及 ``<limit>`` 的 lower/upper/effort/velocity。

支持的 joint type：``revolute``、``continuous``、``prismatic``、``fixed``。
其他类型直接拒绝，不做静默猜测。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .errors import URDFParseError
from .transforms import euler_rpy_to_matrix

_JOINT_TYPES = {"revolute", "continuous", "prismatic", "fixed"}
_JOINT_TYPES_WITH_LIMIT = {"revolute", "prismatic"}


@dataclass(frozen=True)
class Link:
    """URDF link（本工具只需要名字；惯性 / 可视化等标签忽略）。"""

    name: str


@dataclass(frozen=True)
class Origin:
    """``<origin>``：子 joint 框架相对父 link 框架的位姿。"""

    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def matrix(self) -> np.ndarray:
        """4x4 齐次变换矩阵。"""
        t = np.eye(4)
        t[:3, :3] = euler_rpy_to_matrix(self.rpy[0], self.rpy[1], self.rpy[2])
        t[:3, 3] = self.xyz
        return t


@dataclass(frozen=True)
class JointLimit:
    """``<limit>``：单位为弧度（revolute）或米（prismatic）。"""

    lower: float
    upper: float
    effort: float = 0.0
    velocity: float = 0.0

    def clamp(self, value: float) -> float:
        return float(min(self.upper, max(self.lower, value)))


@dataclass
class Joint:
    """URDF joint 的运动学信息。"""

    name: str
    joint_type: str
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    limit: Optional[JointLimit] = None

    @property
    def is_fixed(self) -> bool:
        return self.joint_type == "fixed"

    @property
    def is_continuous(self) -> bool:
        return self.joint_type == "continuous"


@dataclass
class RobotModel:
    """解析得到的原始 URDF 模型（尚未指定基座 / 末端形成运动链）。"""

    name: str
    links: dict[str, Link]
    joints: dict[str, Joint]

    def child_to_parent(self) -> dict[str, Joint]:
        """child link 名 -> 以它为 child 的 joint。"""
        return {joint.child: joint for joint in self.joints.values()}

    def joints_from_parent(self) -> dict[str, list[Joint]]:
        """parent link 名 -> 从它出发的 joint 列表。"""
        out: dict[str, list[Joint]] = {}
        for joint in self.joints.values():
            out.setdefault(joint.parent, []).append(joint)
        return out


def _parse_float3(text: Optional[str], tag: str) -> np.ndarray:
    if text is None or not text.strip():
        return np.zeros(3)
    parts = text.split()
    if len(parts) != 3:
        raise URDFParseError(f"{tag} 必须包含 3 个数字，实际为 {text!r}")
    try:
        return np.array([float(x) for x in parts])
    except ValueError as exc:
        raise URDFParseError(f"{tag} 包含无法解析的数字：{text!r}") from exc


def _parse_origin(element: Optional[ET.Element]) -> Origin:
    if element is None:
        return Origin()
    xyz = _parse_float3(element.get("xyz"), "origin xyz")
    rpy = _parse_float3(element.get("rpy"), "origin rpy")
    return Origin(xyz=xyz, rpy=rpy)


def _parse_axis(element: Optional[ET.Element]) -> np.ndarray:
    if element is None:
        return np.array([0.0, 0.0, 1.0])
    axis = _parse_float3(element.get("xyz"), "axis xyz")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        # 零向量轴在 XML 层面语法合法，但运动学上无意义：
        # 保留原样，由 KinematicChain 构建时抛出 KinematicsError。
        return axis
    # 非零但未归一化的轴在此归一化（URDF 要求单位向量）。
    return axis / norm


def _parse_limit(element: Optional[ET.Element], joint_name: str, joint_type: str) -> Optional[JointLimit]:
    if joint_type not in _JOINT_TYPES_WITH_LIMIT:
        # continuous / fixed：<limit> 可有可无，运动学上不用。
        if element is None:
            return None
        effort = float(element.get("effort", 0.0) or 0.0)
        velocity = float(element.get("velocity", 0.0) or 0.0)
        return JointLimit(lower=-np.inf, upper=np.inf, effort=effort, velocity=velocity)

    if element is None:
        raise URDFParseError(f"joint '{joint_name}' 类型为 {joint_type}，必须提供 <limit>")

    lower_text = element.get("lower")
    upper_text = element.get("upper")
    if lower_text is None or upper_text is None:
        raise URDFParseError(
            f"joint '{joint_name}' 的 <limit> 必须同时提供 lower 和 upper 属性"
        )
    try:
        lower, upper = float(lower_text), float(upper_text)
    except ValueError as exc:
        raise URDFParseError(
            f"joint '{joint_name}' 的 limit lower/upper 必须是数字"
        ) from exc
    if lower >= upper:
        raise URDFParseError(
            f"joint '{joint_name}' 的 limit 非法：要求 lower < upper，"
            f"实际 lower={lower}, upper={upper}"
        )
    effort = float(element.get("effort", 0.0) or 0.0)
    velocity = float(element.get("velocity", 0.0) or 0.0)
    return JointLimit(lower=lower, upper=upper, effort=effort, velocity=velocity)


def parse_urdf(source: str | os.PathLike[str]) -> RobotModel:
    """解析 URDF。

    Parameters
    ----------
    source:
        URDF 文本，或本地 ``.urdf`` / ``.xml`` 文件路径。只读本地文件，
        不发起任何网络请求。
    """
    text = source
    if isinstance(source, (str, os.PathLike)) and "\n" not in str(source) and os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as fh:
            text = fh.read()
    if not isinstance(text, str):
        raise URDFParseError("URDF 源必须是字符串或本地文件路径")
    if not text.strip():
        raise URDFParseError("URDF 内容为空")

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise URDFParseError(f"URDF XML 解析失败：{exc}") from exc
    if root.tag != "robot":
        raise URDFParseError(f"URDF 根元素必须是 <robot>，实际为 <{root.tag}>")

    robot_name = root.get("name", "robot")

    links: dict[str, Link] = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        if not name:
            raise URDFParseError("存在缺少 name 属性的 <link>")
        if name in links:
            raise URDFParseError(f"link 名称重复：'{name}'")
        links[name] = Link(name=name)

    joints: dict[str, Joint] = {}
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        if not name:
            raise URDFParseError("存在缺少 name 属性的 <joint>")
        if name in joints:
            raise URDFParseError(f"joint 名称重复：'{name}'")

        joint_type = joint_el.get("type")
        if joint_type not in _JOINT_TYPES:
            raise URDFParseError(
                f"joint '{name}' 使用了不支持的类型 {joint_type!r}，"
                f"仅支持 {sorted(_JOINT_TYPES)}"
            )

        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or not parent_el.get("link"):
            raise URDFParseError(f"joint '{name}' 缺少 <parent link=...>")
        if child_el is None or not child_el.get("link"):
            raise URDFParseError(f"joint '{name}' 缺少 <child link=...>")
        parent_name, child_name = parent_el.get("link"), child_el.get("link")

        origin = _parse_origin(joint_el.find("origin"))
        axis = _parse_axis(joint_el.find("axis"))
        limit = _parse_limit(joint_el.find("limit"), name, joint_type)

        joints[name] = Joint(
            name=name,
            joint_type=joint_type,
            parent=parent_name,
            child=child_name,
            origin=origin,
            axis=axis,
            limit=limit,
        )

    if not links:
        raise URDFParseError("URDF 中没有任何 <link>")

    # parent / child 指向缺失 link 的检查放在 KinematicChain 构建阶段
    # （那里抛出 KinematicsError）；这里只拒绝明显的自环。
    for joint in joints.values():
        if joint.parent == joint.child:
            raise URDFParseError(
                f"joint '{joint.name}' 的 parent 与 child 不能是同一个 link"
            )

    return RobotModel(name=robot_name, links=links, joints=joints)
