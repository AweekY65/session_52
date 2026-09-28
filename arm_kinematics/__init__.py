"""本地机械臂运动学工具包（纯 NumPy 实现，不依赖 ROS / MoveIt / 硬件 / 外部服务）。

公开接口：
    - urdf_loader:    URDF 解析、链构建与结构校验
    - transforms:     旋转 / 齐次变换 / 位姿误差等基础数学工具
    - kinematics:     正运动学 (FK) 与几何雅可比 (Jacobian)
    - ik:             阻尼最小二乘逆运动学 (IK)
    - finite_diff:    雅可比有限差分验证
"""

from .urdf_loader import (
    URDFError,
    RobotModel,
    Link,
    Joint,
    JointType,
    Origin,
    load_urdf,
    load_urdf_string,
)
from .transforms import (
    rpy_to_matrix,
    make_transform,
    invert_transform,
    rotation_error,
    pose_error,
    rotation_to_axis_angle,
)
from .kinematics import FKResult, forward_kinematics, jacobian
from .finite_diff import finite_difference_jacobian, check_jacobian
from .ik import IKResult, IKStatus, inverse_kinematics, max_reach

__all__ = [
    "URDFError",
    "RobotModel",
    "Link",
    "Joint",
    "JointType",
    "Origin",
    "load_urdf",
    "load_urdf_string",
    "rpy_to_matrix",
    "make_transform",
    "invert_transform",
    "rotation_error",
    "pose_error",
    "rotation_to_axis_angle",
    "FKResult",
    "forward_kinematics",
    "jacobian",
    "finite_difference_jacobian",
    "check_jacobian",
    "IKResult",
    "IKStatus",
    "inverse_kinematics",
    "max_reach",
]
