"""本地机械臂运动学工具（不依赖 ROS / MoveIt / 硬件 / 外部服务）。

模块组成
--------
- :mod:`robot_kinematics.errors`       异常类型
- :mod:`robot_kinematics.transforms`  旋转 / 齐次变换数学工具
- :mod:`robot_kinematics.urdf_parser` URDF 解析（link / joint / origin / axis / limit）
- :mod:`robot_kinematics.chain`       机器人链构建与合法性校验
- :mod:`robot_kinematics.fk`          正向运动学
- :mod:`robot_kinematics.jacobian`    几何雅可比（解析 + 有限差分）
- :mod:`robot_kinematics.ik`          数值逆运动学（阻尼最小二乘）
"""

from .errors import IKError, KinematicsError, URDFParseError
from .chain import KinematicChain
from .fk import forward_kinematics, fk_all
from .ik import (
    IKResult,
    IKStatus,
    inverse_kinematics,
    max_reach,
    within_reach,
)
from .jacobian import finite_difference_jacobian, jacobian
from .transforms import (
    euler_rpy_to_matrix,
    invert_transform,
    matrix_to_rotvec,
    rotvec_to_matrix,
)
from .urdf_parser import Joint, JointLimit, Link, Origin, RobotModel, parse_urdf

__all__ = [
    "URDFParseError",
    "KinematicsError",
    "IKError",
    "Link",
    "Joint",
    "JointLimit",
    "Origin",
    "RobotModel",
    "parse_urdf",
    "KinematicChain",
    "forward_kinematics",
    "fk_all",
    "jacobian",
    "finite_difference_jacobian",
    "IKStatus",
    "IKResult",
    "inverse_kinematics",
    "max_reach",
    "within_reach",
    "euler_rpy_to_matrix",
    "rotvec_to_matrix",
    "matrix_to_rotvec",
    "invert_transform",
]
