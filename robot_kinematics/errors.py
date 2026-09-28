"""运动学工具的异常类型。"""


class URDFParseError(ValueError):
    """URDF 文件本身不合法（XML 错误、缺少必需标签 / 属性、限位非法等）。"""


class KinematicsError(ValueError):
    """模型结构或运动学计算非法（环、缺失 link、非法轴、越限等）。"""


class IKError(ValueError):
    """逆解接口调用方式非法。"""
