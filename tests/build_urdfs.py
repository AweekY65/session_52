"""测试辅助：在工程内动态生成小型机械臂 URDF（2/3/6 关节等）。

不依赖任何外部模型文件或网络，所有模型由本文件的字符串模板生成。
"""

from __future__ import annotations

# 关节类型 -> 是否带有限位
_LIMITED = {"revolute", "prismatic"}


def build_urdf(
    joints: list[dict],
    *,
    robot_name: str = "generated_arm",
    base_link: str = "base_link",
    extra_links: list[str] | None = None,
) -> str:
    """根据关节描述列表生成 URDF 字符串。

    每个 joint dict 的字段：
        name:   关节名（必填）
        type:   revolute / continuous / prismatic / fixed
        parent / child: link 名（可省略，默认按顺序连成链）
        xyz:    <origin xyz>（默认 "0 0 0"）
        rpy:    <origin rpy>（默认 "0 0 0"）
        axis:   <axis xyz>（默认 "1 0 0"，fixed 会忽略）
        lower / upper: 限位（revolute/prismatic 默认 -1.0 / 1.0）
        velocity / effort: 可选
    """
    used_links: list[str] = [base_link]
    xml_joints: list[str] = []
    prev_link = base_link

    for i, spec in enumerate(joints):
        name = spec.get("name", f"joint_{i}")
        jtype = spec["type"]
        parent = spec.get("parent", prev_link)
        child = spec.get("child", f"link_{i}")
        used_links.append(child)
        prev_link = child

        xyz = spec.get("xyz", "0 0 0")
        rpy = spec.get("rpy", "0 0 0")
        lines = [
            f'  <joint name="{name}" type="{jtype}">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{child}"/>',
            f'    <origin xyz="{xyz}" rpy="{rpy}"/>',
        ]
        if jtype != "fixed":
            axis = spec.get("axis", "1 0 0")
            lines.append(f'    <axis xyz="{axis}"/>')
        if jtype in _LIMITED:
            lower = spec.get("lower", -1.0)
            upper = spec.get("upper", 1.0)
            vel = spec.get("velocity", 2.0)
            effort = spec.get("effort", 10.0)
            lines.append(
                f'    <limit lower="{lower}" upper="{upper}" '
                f'effort="{effort}" velocity="{vel}"/>'
            )
        elif jtype == "continuous":
            vel = spec.get("velocity", 2.0)
            effort = spec.get("effort", 10.0)
            lines.append(f'    <limit effort="{effort}" velocity="{vel}"/>')
        lines.append("  </joint>")
        xml_joints.append("\n".join(lines))

    if extra_links:
        used_links.extend(extra_links)

    # 去重并保持顺序
    seen: set[str] = set()
    link_xml = []
    for link_name in used_links:
        if link_name in seen:
            continue
        seen.add(link_name)
        link_xml.append(f'  <link name="{link_name}"/>')

    return (
        '<?xml version="1.0"?>\n'
        f'<robot name="{robot_name}">\n'
        + "\n".join(link_xml)
        + "\n\n"
        + "\n".join(xml_joints)
        + "\n</robot>\n"
    )


# ---------------------------------------------------------------------- #
# 预置小模型
# ---------------------------------------------------------------------- #
def planar_2r_urdf(length1: float = 1.0, length2: float = 1.0,
                   limit: float = 3.14159265358979) -> str:
    """平面 2R 机械臂，两关节均绕基座 z 轴。

    结构（符合 URDF 语义：child link 框架位于其上游关节轴处）::

        base_link --j1--> l1 --j2(origin x=l1)--> l2 --j_ee(固定, xyz=l2)--> end_effector
    """
    return build_urdf(
        robot_name="planar_2r",
        joints=[
            {"name": "j1", "type": "revolute", "child": "l1",
             "axis": "0 0 1", "lower": -limit, "upper": limit},
            {"name": "j2", "type": "revolute", "child": "l2",
             "xyz": f"{length1} 0 0", "axis": "0 0 1",
             "lower": -limit, "upper": limit},
            {"name": "j_ee", "type": "fixed", "child": "end_effector",
             "xyz": f"{length2} 0 0"},
        ],
    )


def three_joint_mixed_urdf() -> str:
    """3 关节模型：revolute(y) + fixed 偏移 + continuous(z) + prismatic(z)。

    实际有 3 个活动关节（fixed 不占自由度）。
    """
    return build_urdf(
        robot_name="mixed_3",
        joints=[
            {"name": "j_rev", "type": "revolute", "child": "l1",
             "xyz": "0 0 0.1", "axis": "0 1 0",
             "lower": -1.2, "upper": 1.2},
            {"name": "j_fix", "type": "fixed", "child": "l2",
             "xyz": "0.3 0 0", "rpy": "0 0 1.5707963267948966"},
            {"name": "j_cont", "type": "continuous", "child": "l3",
             "axis": "0 0 1"},
            {"name": "j_prism", "type": "prismatic", "child": "l4",
             "xyz": "0.2 0 0", "axis": "0 0 1",
             "lower": -0.25, "upper": 0.25},
        ],
    )


def six_revolute_urdf() -> str:
    """6 关节全 revolute 模型（轴交替、带来源偏移，用于一般 6R 测试）。

    相邻关节轴尽量不共线，使零位雅可比满秩，避免零位恰好落在腕部/肩部
    奇异上（IK 的奇异处理由其他用例覆盖）。
    """
    specs = [
        ("j1", "0 0 0.1", "0 0 0", "0 0 1", -3.14, 3.14),
        ("j2", "0 0 0.3", "0 0 0", "0 1 0", -1.9, 1.9),
        ("j3", "0.05 0 0.4", "0 0 0", "0 0 1", -2.5, 2.5),
        ("j4", "0.4 0 0", "0 0.7853981633974483 0", "0 1 0", -3.14, 3.14),
        ("j5", "0 0 0.05", "0 -0.7853981633974483 0", "0 0 1", -2.0, 2.0),
        ("j6", "0 0 0.08", "0 0 0", "0 1 0", -6.28, 6.28),
    ]
    joints = [
        {"name": n, "type": "revolute", "child": f"l{i + 1}",
         "xyz": xyz, "rpy": rpy, "axis": axis,
         "lower": lo, "upper": hi}
        for i, (n, xyz, rpy, axis, lo, hi) in enumerate(specs)
    ]
    return build_urdf(robot_name="six_r", joints=joints)
