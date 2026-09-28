"""URDF 解析与链构建测试：正常解析 + 循环 / 缺失 link / 非法轴等错误检测。"""

from __future__ import annotations

import numpy as np
import pytest

from arm_kinematics import URDFError, JointType, load_urdf_string

from build_urdfs import build_urdf, planar_2r_urdf


def test_parse_basic_chain_fields():
    robot = load_urdf_string(planar_2r_urdf(0.7, 0.9))
    assert robot.name == "planar_2r"
    assert robot.base_link == "base_link"
    assert robot.end_link == "end_effector"
    assert robot.link_names() == ["base_link", "l1", "l2", "end_effector"]
    assert [j.name for j in robot.chain] == ["j1", "j2", "j_ee"]
    assert robot.nq == 2
    assert robot.joint_limits() == [
        pytest.approx((-np.pi, np.pi)),
        pytest.approx((-np.pi, np.pi)),
    ]
    j2 = robot.joints["j2"]
    assert np.allclose(j2.origin.xyz, [0.7, 0, 0])
    assert np.allclose(j2.axis, [0, 0, 1])
    assert j2.joint_type is JointType.REVOLUTE
    # 末端固定关节的杆长偏移
    jee = robot.joints["j_ee"]
    assert jee.joint_type is JointType.FIXED
    assert np.allclose(jee.origin.xyz, [0.9, 0, 0])


def test_parse_all_joint_types_and_axis_normalization():
    urdf = build_urdf([
        {"name": "jr", "type": "revolute", "child": "l1", "axis": "0 2 0"},
        {"name": "jf", "type": "fixed", "child": "l2"},
        {"name": "jc", "type": "continuous", "child": "l3", "axis": "0 0 -1"},
        {"name": "jp", "type": "prismatic", "child": "l4",
         "axis": "1 1 0", "lower": -0.1, "upper": 0.1},
    ])
    robot = load_urdf_string(urdf)
    types = [j.joint_type for j in robot.chain]
    assert types == [JointType.REVOLUTE, JointType.FIXED,
                     JointType.CONTINUOUS, JointType.PRISMATIC]
    # 轴被归一化
    assert np.allclose(robot.joints["jr"].axis, [0, 1, 0])
    assert np.allclose(robot.joints["jc"].axis, [0, 0, -1])
    assert np.allclose(robot.joints["jp"].axis, [1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    # continuous 无限位
    assert robot.joints["jc"].lower is None
    # 活动关节只有 3 个，fixed 不占 q
    assert robot.nq == 3


def test_missing_parent_link_detected():
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "parent": "ghost", "child": "l1"},
    ])
    with pytest.raises(URDFError, match="parent link"):
        load_urdf_string(urdf)


def test_missing_child_link_name_still_defined_case():
    """joint 的 child 名称根本没出现在 <link> 中"""
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "declared_l1"},
    ])
    # build_urdf 会声明 child link，这里手工构造一个不声明的情形
    bad = urdf.replace('<link name="declared_l1"/>', "")
    with pytest.raises(URDFError, match="child link"):
        load_urdf_string(bad)


def test_cycle_via_two_parents_detected():
    """l1 被两个关节当作 child —— 等价于汇合/循环，解析阶段拒绝。"""
    urdf = """
    <robot name="cyc">
      <link name="base_link"/>
      <link name="l1"/>
      <link name="l2"/>
      <joint name="a" type="fixed">
        <parent link="base_link"/><child link="l1"/>
      </joint>
      <joint name="b" type="fixed">
        <parent link="l2"/><child link="l1"/>
      </joint>
      <joint name="c" type="fixed">
        <parent link="l1"/><child link="l2"/>
      </joint>
    </robot>"""
    with pytest.raises(URDFError, match="汇合/循环"):
        load_urdf_string(urdf)


def test_self_loop_detected():
    urdf = """
    <robot name="self">
      <link name="base_link"/>
      <joint name="j" type="revolute">
        <parent link="base_link"/><child link="base_link"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1" upper="1"/>
      </joint>
    </robot>"""
    with pytest.raises(URDFError, match="自环"):
        load_urdf_string(urdf)


def test_zero_axis_rejected():
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1", "axis": "0 0 0"},
    ])
    with pytest.raises(URDFError, match="零向量"):
        load_urdf_string(urdf)


def test_unsupported_joint_type_rejected():
    urdf = build_urdf([
        {"name": "j1", "type": "floating", "child": "l1"},
    ])
    with pytest.raises(URDFError, match="不受支持"):
        load_urdf_string(urdf)


def test_missing_limit_rejected():
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1"},
    ]).replace('    <limit lower="-1.0" upper="1.0" effort="10.0" velocity="2.0"/>\n', "")
    with pytest.raises(URDFError, match="limit"):
        load_urdf_string(urdf)


def test_inverted_limit_rejected():
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1",
         "lower": 1.0, "upper": -1.0},
    ])
    with pytest.raises(URDFError, match="lower"):
        load_urdf_string(urdf)


def test_duplicate_link_and_joint_names():
    urdf = """
    <robot name="dup">
      <link name="base_link"/>
      <link name="base_link"/>
      <joint name="j1" type="fixed">
        <parent link="base_link"/><child link="base_link"/>
      </joint>
    </robot>"""
    with pytest.raises(URDFError, match="重复"):
        load_urdf_string(urdf)


def test_multiple_roots_rejected():
    urdf = """
    <robot name="roots">
      <link name="a"/>
      <link name="b"/>
    </robot>"""
    with pytest.raises(URDFError, match="根 link"):
        load_urdf_string(urdf)


def test_multiple_leaves_requires_explicit_end():
    urdf = build_urdf([
        {"name": "j1", "type": "fixed", "child": "l1"},
        {"name": "j2", "type": "fixed", "parent": "base_link", "child": "l2"},
    ])
    with pytest.raises(URDFError, match="多个末端"):
        load_urdf_string(urdf)
    # 显式指定 end_link 后可以正常构建，且另一条分支被排除
    robot = load_urdf_string(urdf, end_link="l2")
    assert robot.link_names() == ["base_link", "l2"]


def test_end_link_not_downstream_of_base():
    urdf = build_urdf([
        {"name": "j1", "type": "fixed", "child": "l1"},
        {"name": "j2", "type": "fixed", "parent": "l1", "child": "l2"},
    ])
    with pytest.raises(URDFError, match="不在 base_link"):
        load_urdf_string(urdf, base_link="l2", end_link="base_link")


def test_unknown_base_or_end_link():
    robot_urdf = planar_2r_urdf()
    with pytest.raises(URDFError, match="base_link"):
        load_urdf_string(robot_urdf, base_link="nope")
    with pytest.raises(URDFError, match="end_link"):
        load_urdf_string(robot_urdf, end_link="nope")


def test_bad_xml_rejected():
    with pytest.raises(URDFError, match="XML"):
        load_urdf_string("<robot><link></robot>")


def test_subchain_extract():
    """分支结构上选取不同 end_link 得到不同子链。"""
    urdf = build_urdf([
        {"name": "j1", "type": "revolute", "child": "l1", "axis": "0 0 1"},
        {"name": "j2a", "type": "revolute", "parent": "l1",
         "child": "la", "axis": "0 0 1"},
        {"name": "j2b", "type": "fixed", "parent": "l1", "child": "lb"},
    ])
    full = load_urdf_string(urdf, end_link="la")
    assert full.link_names() == ["base_link", "l1", "la"]
    sub = full.extract_chain(end_link="lb")
    assert sub.link_names() == ["base_link", "l1", "lb"]
    assert sub.nq == 1
