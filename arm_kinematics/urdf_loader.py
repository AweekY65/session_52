"""URDF 解析与机器人运动链构建。

仅使用 Python 标准库的 XML 解析器读取本地 URDF 文件，解析内容包括：
    - <link> / <joint> 的名称与层级关系
    - <origin xyz rpy>
    - <axis xyz>（缺省为 (1, 0, 0)）
    - 关节类型：revolute / continuous / prismatic / fixed
    - 关节限位 <limit lower upper>（revolute / prismatic 必填）

结构校验：
    - 缺失 link（joint 指向不存在的 parent/child）
    - 重复 link / joint 名称、重复父子关系
    - 非法关节轴（零向量或非有限值）
    - 不支持的关节类型（floating / planar 等）
    - 限位缺失或 lower > upper
    - 运动链中的循环结构（joint 构成的有向图成环）
    - 多个根 link，或指定的 base_link / end_link 不存在
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET

import numpy as np


class URDFError(ValueError):
    """URDF 解析或运动链构建失败时抛出。"""


class JointType(str, Enum):
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FIXED = "fixed"

    @property
    def movable(self) -> bool:
        return self is not JointType.FIXED


@dataclass(frozen=True)
class Origin:
    """<origin>：相对父关节（父 link）的平移 xyz 与固定角 rpy。"""

    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Link:
    name: str


@dataclass
class Joint:
    name: str
    joint_type: JointType
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None

    @property
    def movable(self) -> bool:
        return self.joint_type.movable


def _parse_vec(text: str | None, default: str) -> np.ndarray:
    """解析 URDF 中的三元素向量字符串，非法时抛出 URDFError。"""
    text = (text or "").strip() or default
    try:
        values = [float(v) for v in text.split()]
    except ValueError as exc:
        raise URDFError(f"无法解析向量 {text!r}：{exc}") from exc
    if len(values) != 3:
        raise URDFError(f"向量必须包含 3 个数字，实际为 {text!r}")
    arr = np.array(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise URDFError(f"向量包含非有限值：{text!r}")
    return arr


@dataclass
class RobotModel:
    """已校验的机器人模型及其选定运动链（base_link -> end_link）。

    Attributes:
        name: <robot name>
        links / joints: URDF 中解析出的全部 link / joint
        chain: 从 base 到 end 路径上的关节（含 fixed），按基座到末端排序
        movable_joints: chain 中去掉 fixed 后的关节，顺序即广义坐标 q 的顺序
        base_link / end_link: 运动链两端 link 名称
    """

    name: str
    links: dict[str, Link]
    joints: dict[str, Joint]
    chain: list[Joint]
    base_link: str
    end_link: str

    # ------------------------------------------------------------------ #
    @property
    def movable_joints(self) -> list[Joint]:
        return [j for j in self.chain if j.movable]

    @property
    def nq(self) -> int:
        """广义坐标（活动关节）数量。"""
        return len(self.movable_joints)

    def link_names(self) -> list[str]:
        """沿运动链从基座到末端的所有 link 名称（含 base 与 end）。"""
        names = [self.base_link]
        names.extend(j.child for j in self.chain)
        return names

    def joint_limits(self) -> list[tuple[float, float]]:
        """各活动关节的 (lower, upper)；continuous 返回 (-inf, inf)。"""
        limits: list[tuple[float, float]] = []
        for j in self.movable_joints:
            if j.joint_type is JointType.CONTINUOUS:
                limits.append((-np.inf, np.inf))
            else:
                assert j.lower is not None and j.upper is not None
                limits.append((j.lower, j.upper))
        return limits

    def extract_chain(self, base_link: str | None = None,
                      end_link: str | None = None) -> "RobotModel":
        """在同一已解析 URDF 上重新选取 base/end 并构建子链（返回新模型对象）。"""
        return _build_chain(self.name, self.links, self.joints,
                            base_link, end_link)


# ---------------------------------------------------------------------- #
# 解析
# ---------------------------------------------------------------------- #
_SUPPORTED_TYPES = {t.value for t in JointType}


def _parse_joint(elem: ET.Element, links: dict[str, Link]) -> Joint:
    name = elem.get("name")
    if not name:
        raise URDFError("存在缺少 name 属性的 <joint>")
    type_str = elem.get("type")
    if type_str not in _SUPPORTED_TYPES:
        raise URDFError(
            f"关节 {name!r} 的类型 {type_str!r} 不受支持；"
            f"仅支持 {sorted(_SUPPORTED_TYPES)}"
        )
    joint_type = JointType(type_str)

    parent_elem = elem.find("parent")
    child_elem = elem.find("child")
    if parent_elem is None or not parent_elem.get("link"):
        raise URDFError(f"关节 {name!r} 缺少 <parent link=...>")
    if child_elem is None or not child_elem.get("link"):
        raise URDFError(f"关节 {name!r} 缺少 <child link=...>")
    parent_name = parent_elem.get("link")
    child_name = child_elem.get("link")

    if parent_name not in links:
        raise URDFError(f"关节 {name!r} 的 parent link {parent_name!r} 未在 URDF 中定义")
    if child_name not in links:
        raise URDFError(f"关节 {name!r} 的 child link {child_name!r} 未在 URDF 中定义")
    if parent_name == child_name:
        raise URDFError(f"关节 {name!r} 的 parent 与 child 同名（{parent_name!r}），构成自环")

    origin_elem = elem.find("origin")
    if origin_elem is not None:
        origin = Origin(
            xyz=_parse_vec(origin_elem.get("xyz"), "0 0 0"),
            rpy=_parse_vec(origin_elem.get("rpy"), "0 0 0"),
        )
    else:
        origin = Origin()

    axis = np.array([1.0, 0.0, 0.0])
    axis_elem = elem.find("axis")
    if axis_elem is not None:
        axis = _parse_vec(axis_elem.get("xyz"), "1 0 0")
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            raise URDFError(f"关节 {name!r} 的 axis 为零向量，旋转/平移轴无定义")
        axis = axis / norm

    lower = upper = effort = velocity = None
    limit_elem = elem.find("limit")
    needs_limits = joint_type in (JointType.REVOLUTE, JointType.PRISMATIC)
    if needs_limits:
        if limit_elem is None:
            raise URDFError(
                f"关节 {name!r}（{type_str}）缺少 <limit lower upper>"
            )
        lower_attr = limit_elem.get("lower")
        upper_attr = limit_elem.get("upper")
        if lower_attr is None or upper_attr is None:
            raise URDFError(
                f"关节 {name!r} 的 <limit> 必须同时给出 lower 与 upper"
            )
        lower = float(lower_attr)
        upper = float(upper_attr)
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise URDFError(f"关节 {name!r} 的限位必须为有限值")
        if lower > upper:
            raise URDFError(
                f"关节 {name!r} 的限位非法：lower({lower}) > upper({upper})"
            )
        effort = _optional_float(limit_elem.get("effort"))
        velocity = _optional_float(limit_elem.get("velocity"))
    elif limit_elem is not None:
        # continuous / fixed 也可能带 effort/velocity（continuous 常带 velocity）
        effort = _optional_float(limit_elem.get("effort"))
        velocity = _optional_float(limit_elem.get("velocity"))

    if elem.find("mimic") is not None:
        raise URDFError(f"关节 {name!r} 使用了 <mimic>，本工具暂不支持")

    return Joint(
        name=name,
        joint_type=joint_type,
        parent=parent_name,
        child=child_name,
        origin=origin,
        axis=axis,
        lower=lower,
        upper=upper,
        effort=effort,
        velocity=velocity,
    )


def _optional_float(text: str | None) -> float | None:
    if text is None:
        return None
    return float(text)


def _parse_tree(root: ET.Element) -> tuple[str, dict[str, Link], dict[str, Joint]]:
    if root.tag != "robot":
        raise URDFError("URDF 根节点必须是 <robot>")
    robot_name = root.get("name") or "robot"

    links: dict[str, Link] = {}
    for elem in root.findall("link"):
        name = elem.get("name")
        if not name:
            raise URDFError("存在缺少 name 属性的 <link>")
        if name in links:
            raise URDFError(f"link 名称重复：{name!r}")
        links[name] = Link(name=name)

    if not links:
        raise URDFError("URDF 中没有任何 <link>")

    joints: dict[str, Joint] = {}
    children_used: dict[str, str] = {}
    for elem in root.findall("joint"):
        joint = _parse_joint(elem, links)
        if joint.name in joints:
            raise URDFError(f"joint 名称重复：{joint.name!r}")
        if joint.child in children_used:
            raise URDFError(
                f"link {joint.child!r} 同时是关节 {children_used[joint.child]!r} "
                f"与 {joint.name!r} 的 child，存在汇合/循环结构（树中每个 link 只能有一个父关节）"
            )
        children_used[joint.child] = joint.name
        joints[joint.name] = joint

    return robot_name, links, joints


# ---------------------------------------------------------------------- #
# 运动链构建
# ---------------------------------------------------------------------- #
def _build_chain(
    name: str,
    links: dict[str, Link],
    joints: dict[str, Joint],
    base_link: str | None,
    end_link: str | None,
) -> RobotModel:
    # child -> joint，构建有向树
    child_to_joint: dict[str, Joint] = {}
    parent_to_joints: dict[str, list[Joint]] = {}
    for j in joints.values():
        child_to_joint[j.child] = j
        parent_to_joints.setdefault(j.parent, []).append(j)

    root_links = [link_name for link_name in links if link_name not in child_to_joint]
    if len(root_links) != 1:
        raise URDFError(
            f"URDF 必须恰好有 1 个根 link（没有任何父关节），实际找到 {len(root_links)} 个："
            f"{root_links}"
        )
    root_name = root_links[0]

    # 环检测：joint 有向图中每个节点最多一个父关节已在解析阶段保证；
    # 这里再沿 parent 指针向上追溯，确认必然终止于根，防御性检测循环。
    for link_name in links:
        seen: set[str] = set()
        cur = link_name
        while cur in child_to_joint:
            if cur in seen:
                raise URDFError(f"运动链中检测到循环结构，涉及 link {cur!r}")
            seen.add(cur)
            cur = child_to_joint[cur].parent

    if base_link is None:
        base_link = root_name
    if end_link is None:
        leaves = [
            link_name for link_name in links
            if link_name not in parent_to_joints or not parent_to_joints[link_name]
        ]
        if len(leaves) == 1:
            end_link = leaves[0]
        elif base_link in leaves:
            end_link = base_link
        else:
            raise URDFError(
                f"URDF 有多个末端 link {leaves}，请显式指定 end_link"
            )

    if base_link not in links:
        raise URDFError(f"base_link {base_link!r} 不存在于 URDF 中")
    if end_link not in links:
        raise URDFError(f"end_link {end_link!r} 不存在于 URDF 中")

    # 从 end_link 沿父关节回溯到 base_link，得到 base -> end 的关节序列
    reversed_chain: list[Joint] = []
    cur = end_link
    reached = False
    guard = 0
    while True:
        if cur == base_link:
            reached = True
            break
        if cur not in child_to_joint:
            break
        reversed_chain.append(child_to_joint[cur])
        cur = child_to_joint[cur].parent
        guard += 1
        if guard > len(joints) + 1:
            raise URDFError("回溯运动链时异常终止，疑似存在循环结构")
    if not reached:
        raise URDFError(
            f"end_link {end_link!r} 不在 base_link {base_link!r} 的下游，"
            f"无法构成单一运动链（最近回溯到 {cur!r}）"
        )

    chain = list(reversed(reversed_chain))
    return RobotModel(
        name=name,
        links=links,
        joints=joints,
        chain=chain,
        base_link=base_link,
        end_link=end_link,
    )


def load_urdf_string(xml_text: str, base_link: str | None = None,
                     end_link: str | None = None) -> RobotModel:
    """从 URDF XML 字符串构建机器人模型。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise URDFError(f"URDF XML 解析失败：{exc}") from exc
    name, links, joints = _parse_tree(root)
    return _build_chain(name, links, joints, base_link, end_link)


def load_urdf(path: str | os.PathLike, base_link: str | None = None,
              end_link: str | None = None) -> RobotModel:
    """从本地 URDF 文件路径构建机器人模型。"""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise URDFError(f"URDF 文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return load_urdf_string(f.read(), base_link=base_link,
                                end_link=end_link)
