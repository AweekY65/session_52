"""测试用 URDF 模型工厂：在工程内程序化生成小型 2/3/6 关节模型。

所有模型都是纯字符串，不依赖任何外部文件或 ROS 资源。
"""

from __future__ import annotations

import numpy as np


def planar_arm_urdf(n_joints: int, lengths, joint_limit: float = np.pi) -> str:
    """经典平面 nR 机械臂：关节 i 位于连杆 i 起点，全部绕 z 轴旋转。

    结构：base --j0(origin 0)--> l0 --j1(origin L0)--> l1 ... --fixed tip(origin L_{n-1})
    因此末端位置满足教科书公式
    ``p = sum_i L_i * [cos(sum_{k<=i} q_k), sin(sum_{k<=i} q_k)]``。
    """
    lengths = [float(x) for x in lengths]
    assert len(lengths) == n_joints
    parts = ['<robot name="planar">', '<link name="base"/>']
    parent = "base"
    for i in range(n_joints):
        child = f"l{i}"
        origin_x = 0.0 if i == 0 else lengths[i - 1]
        parts.append(f'<link name="{child}"/>')
        parts.append(
            f'<joint name="j{i}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{origin_x} 0 0" rpy="0 0 0"/>'
            f'<axis xyz="0 0 1"/>'
            f'<limit lower="{-joint_limit}" upper="{joint_limit}" '
            f'effort="10" velocity="2"/></joint>'
        )
        parent = child
    parts.append('<link name="tip"/>')
    parts.append(
        f'<joint name="tip_fixed" type="fixed">'
        f'<parent link="{parent}"/><child link="tip"/>'
        f'<origin xyz="{lengths[-1]} 0 0"/></joint>'
    )
    parts.append("</robot>")
    return "\n".join(parts)


def arm_6r_urdf() -> str:
    """KUKA 风格 6R 机械臂（球腕），用于全 6 自由度位姿 IK。"""
    joints = [
        # name, parent, child, origin xyz, axis
        ("j1", "base", "l1", "0 0 0.35", "0 0 1"),
        ("j2", "l1", "l2", "0 0 0", "0 1 0"),
        ("j3", "l2", "l3", "0 0 0.40", "0 1 0"),
        ("j4", "l3", "l4", "0 0 0", "0 0 1"),
        ("j5", "l4", "l5", "0 0 0.39", "0 1 0"),
        ("j6", "l5", "tip", "0 0 0.10", "0 0 1"),
    ]
    parts = ['<robot name="arm6r">', '<link name="base"/>']
    for name, parent, child, origin, axis in joints:
        parts.append(f'<link name="{child}"/>')
        parts.append(
            f'<joint name="{name}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{origin}"/><axis xyz="{axis}"/>'
            f'<limit lower="-3.14159" upper="3.14159" effort="100" velocity="2"/>'
            f"</joint>"
        )
    parts.append("</robot>")
    return "\n".join(parts)


def prismatic_urdf(lower: float = -0.5, upper: float = 1.0) -> str:
    """单自由度 prismatic 滑台，沿 x 轴。"""
    return f"""<robot name="slider">
  <link name="base"/>
  <link name="tip"/>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="tip"/>
    <origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="{lower}" upper="{upper}" effort="10" velocity="1"/>
  </joint>
</robot>"""


def fixed_only_urdf() -> str:
    """零自由度链：base 与 tip 之间只有一个 fixed joint。"""
    return """<robot name="fixed_only">
  <link name="base"/>
  <link name="tip"/>
  <joint name="weld" type="fixed">
    <parent link="base"/><child link="tip"/>
    <origin xyz="0.3 0.1 0" rpy="0 0 0.5"/>
  </joint>
</robot>"""


def limited_2r_urdf(j2_limit: float = 0.1) -> str:
    """2R 平面臂，第二关节限位极小，用于「目标被限位挡住」场景。"""
    return f"""<robot name="limited2r">
  <link name="base"/><link name="l0"/><link name="l1"/>
  <joint name="j0" type="revolute">
    <parent link="base"/><child link="l0"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <joint name="j1" type="revolute">
    <parent link="l0"/><child link="l1"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="{-j2_limit}" upper="{j2_limit}" effort="10" velocity="1"/>
  </joint>
  <joint name="tip_fixed" type="fixed">
    <parent link="l1"/><child link="tip"/>
    <origin xyz="1 0 0"/>
  </joint>
  <link name="tip"/>
</robot>"""
