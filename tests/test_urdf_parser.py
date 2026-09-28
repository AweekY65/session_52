"""URDF 解析与运动链构建的测试。"""

import numpy as np
import pytest

from robot_kinematics import (
    KinematicChain,
    KinematicsError,
    URDFParseError,
    parse_urdf,
)
from robot_kinematics.transforms import euler_rpy_to_matrix

from .urdfs import fixed_only_urdf, planar_arm_urdf


def build_chain(urdf_text: str, **kwargs) -> KinematicChain:
    return KinematicChain(parse_urdf(urdf_text), **kwargs)


# ---------------------------------------------------------------------- #
# 正常解析
# ---------------------------------------------------------------------- #
def test_parse_links_joints_origin_axis_limit():
    model = parse_urdf(planar_arm_urdf(2, [1.0, 0.7]))
    assert set(model.links) == {"base", "l0", "l1", "tip"}
    assert set(model.joints) == {"j0", "j1", "tip_fixed"}

    j1 = model.joints["j1"]
    assert j1.joint_type == "revolute"
    assert j1.parent == "l0" and j1.child == "l1"
    np.testing.assert_allclose(j1.origin.xyz, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(j1.origin.rpy, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(j1.axis, [0.0, 0.0, 1.0])
    assert j1.limit is not None
    assert j1.limit.lower == pytest.approx(-np.pi)
    assert j1.limit.upper == pytest.approx(np.pi)
    assert j1.limit.effort == pytest.approx(10.0)
    assert j1.limit.velocity == pytest.approx(2.0)

    fixed = model.joints["tip_fixed"]
    assert fixed.is_fixed
    np.testing.assert_allclose(fixed.origin.xyz, [0.7, 0.0, 0.0])


def test_origin_rpy_uses_urdf_convention():
    # URDF origin rpy 是固定轴 XYZ（外旋）：R = Rz(yaw) Ry(pitch) Rx(roll)
    urdf = """<robot name="r">
      <link name="base"/><link name="tip"/>
      <joint name="j" type="fixed">
        <parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0" rpy="0.1 0.2 0.3"/>
      </joint>
    </robot>"""
    model = parse_urdf(urdf)
    expected = euler_rpy_to_matrix(0.1, 0.2, 0.3)
    np.testing.assert_allclose(model.joints["j"].origin.matrix[:3, :3], expected, atol=1e-12)
    # 与手算 Rz(0.3) @ Ry(0.2) @ Rx(0.1) 一致
    from robot_kinematics.transforms import rx_matrix, ry_matrix, rz_matrix

    np.testing.assert_allclose(
        expected, rz_matrix(0.3) @ ry_matrix(0.2) @ rx_matrix(0.1), atol=1e-12
    )


def test_non_unit_axis_is_normalized():
    urdf = """<robot name="r">
      <link name="base"/><link name="tip"/>
      <joint name="j" type="continuous">
        <parent link="base"/><child link="tip"/>
        <axis xyz="0 0 2.5"/>
      </joint>
    </robot>"""
    model = parse_urdf(urdf)
    np.testing.assert_allclose(model.joints["j"].axis, [0.0, 0.0, 1.0])


def test_continuous_joint_has_infinite_limits():
    urdf = """<robot name="r">
      <link name="base"/><link name="tip"/>
      <joint name="spin" type="continuous">
        <parent link="base"/><child link="tip"/>
        <axis xyz="0 0 1"/>
      </joint>
    </robot>"""
    chain = build_chain(urdf)
    lo, hi = chain.joint_limits[0]
    assert lo == -np.inf and hi == np.inf


def test_chain_order_and_fixed_joints_excluded_from_dof():
    chain = build_chain(planar_arm_urdf(3, [1.0, 0.8, 0.6]))
    assert chain.base_link == "base"
    assert chain.tip_link == "tip"
    assert chain.link_names == ["base", "l0", "l1", "l2", "tip"]
    assert [j.name for j in chain.joints] == ["j0", "j1", "j2", "tip_fixed"]
    assert chain.joint_names == ["j0", "j1", "j2"]
    assert chain.dof == 3  # fixed joint 不计入自由度


# ---------------------------------------------------------------------- #
# 结构错误
# ---------------------------------------------------------------------- #
def test_duplicate_link_name():
    urdf = """<robot name="r">
      <link name="a"/><link name="a"/>
    </robot>"""
    with pytest.raises(URDFParseError, match="重复"):
        parse_urdf(urdf)


def test_bad_xml_and_empty():
    with pytest.raises(URDFParseError):
        parse_urdf("this is not xml <><>")
    with pytest.raises(URDFParseError):
        parse_urdf("")
    with pytest.raises(URDFParseError):
        parse_urdf("   \n  ")
    with pytest.raises(URDFParseError, match="robot"):
        parse_urdf("<notrobot/>")


def test_unsupported_joint_type_rejected():
    urdf = """<robot name="r">
      <link name="a"/><link name="b"/>
      <joint name="j" type="floating">
        <parent link="a"/><child link="b"/>
      </joint>
    </robot>"""
    with pytest.raises(URDFParseError, match="不支持"):
        parse_urdf(urdf)


def test_disconnected_tree_requires_base():
    # 两条互不相连的链：不显式给 base_link 必须报错
    urdf = """<robot name="r">
      <link name="a0"/><link name="a1"/>
      <link name="b0"/><link name="b1"/>
      <joint name="ja" type="fixed"><parent link="a0"/><child link="a1"/></joint>
      <joint name="jb" type="fixed"><parent link="b0"/><child link="b1"/></joint>
    </robot>"""
    with pytest.raises(KinematicsError, match="base_link"):
        build_chain(urdf)
    chain = build_chain(urdf, base_link="b0")
    assert chain.base_link == "b0" and chain.tip_link == "b1"


def test_multiple_leaves_requires_tip():
    urdf = """<robot name="r">
      <link name="base"/><link name="x"/><link name="y"/>
      <joint name="jx" type="fixed"><parent link="base"/><child link="x"/></joint>
      <joint name="jy" type="fixed"><parent link="base"/><child link="y"/></joint>
    </robot>"""
    with pytest.raises(KinematicsError, match="tip_link"):
        build_chain(urdf)
    chain = build_chain(urdf, tip_link="y")
    assert chain.tip_link == "y"


# ---------------------------------------------------------------------- #
# 参数化的非法模型
# ---------------------------------------------------------------------- #
BAD_AXIS_URDF = """<robot name="r">
  <link name="a"/><link name="b"/>
  <joint name="j" type="revolute">
    <parent link="a"/><child link="b"/>
    <axis xyz="0 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""

BAD_LIMIT_URDF = """<robot name="r">
  <link name="a"/><link name="b"/>
  <joint name="j" type="revolute">
    <parent link="a"/><child link="b"/>
    <axis xyz="0 0 1"/>
    <limit lower="1.0" upper="-1.0" effort="1" velocity="1"/>
  </joint>
</robot>"""

CYCLIC_URDF = """<robot name="r">
  <link name="a"/><link name="b"/><link name="c"/>
  <joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>
  <joint name="j2" type="fixed"><parent link="b"/><child link="c"/></joint>
  <joint name="j3" type="fixed"><parent link="c"/><child link="a"/></joint>
</robot>"""

MISSING_LIMIT_URDF = """<robot name="r">
  <link name="a"/><link name="b"/>
  <joint name="j" type="revolute">
    <parent link="a"/><child link="b"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>"""

MISSING_LINK_URDF = """<robot name="r">
  <link name="a"/>
  <joint name="j" type="fixed"><parent link="a"/><child link="ghost"/></joint>
</robot>"""

MULTI_PARENT_URDF = """<robot name="r">
  <link name="a"/><link name="b"/><link name="c"/>
  <joint name="j1" type="fixed"><parent link="a"/><child link="c"/></joint>
  <joint name="j2" type="fixed"><parent link="b"/><child link="c"/></joint>
</robot>"""

INVALID_URDFS = {
    "bad_axis_urdf": BAD_AXIS_URDF,
    "bad_limit_urdf": BAD_LIMIT_URDF,
    "cyclic_urdf": CYCLIC_URDF,
    "missing_limit_urdf": MISSING_LIMIT_URDF,
    "missing_link_urdf": MISSING_LINK_URDF,
    "multi_parent_urdf": MULTI_PARENT_URDF,
}


@pytest.mark.parametrize(
    "name,exc_type,keyword",
    [
        ("bad_axis_urdf", KinematicsError, "零向量"),
        ("bad_limit_urdf", URDFParseError, "lower"),
        ("cyclic_urdf", KinematicsError, "环"),
        ("missing_limit_urdf", URDFParseError, "limit"),
        ("missing_link_urdf", KinematicsError, "不存在"),
        ("multi_parent_urdf", KinematicsError, "父"),
    ],
)
def test_invalid_models_raise(name, exc_type, keyword):
    with pytest.raises(exc_type, match=keyword):
        build_chain(INVALID_URDFS[name])


def test_parse_from_file(tmp_path):
    path = tmp_path / "arm.urdf"
    path.write_text(planar_arm_urdf(2, [1.0, 0.5]), encoding="utf-8")
    model = parse_urdf(str(path))
    assert model.name == "planar"
    assert set(model.joints) == {"j0", "j1", "tip_fixed"}
