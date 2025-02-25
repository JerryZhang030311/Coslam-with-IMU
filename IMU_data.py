from rotation_conversions import Lie
import torch
import numpy as np
import torch.nn.functional as F

# 创建 Lie 类的实例，用于处理 Lie 群相关的旋转和位移操作
lie_instance = Lie()


def matrix_logarithm(R):
    """
    Compute the matrix logarithm of a 3x3 rotation matrix R.
    计算 3x3 旋转矩阵 R 的矩阵对数，返回对应的旋转轴。
    """
    # Ensure R is a rotation matrix
    # 确保 R 是一个旋转矩阵
    R = F.normalize(R, dim=(0, 1))

    # 计算旋转角度
    angle = torch.acos((torch.trace(R) - 1) / 2)

    if angle < 1e-5:
        # 如果角度接近 0（接近单位矩阵），使用一阶近似
        omega = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    else:
        # 一般情况下，使用标准公式计算旋转轴
        omega = angle / (2 * torch.sin(angle)) * torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])

    return omega


def interpolate_se3(se3a, se3b, new_time):
    """
    Interpolate between two SE(3) transformations using spherical linear interpolation (SLERP).
    使用球面线性插值（SLERP）在两个 SE(3) 变换之间进行插值。
    """
    assert se3a.shape == se3b.shape, "not the same shape!"
    assert len(se3a.shape) == 1, "cannot interpolate!"

    # 归一化输入的 SE(3) 向量
    se3a = se3a / torch.norm(se3a)
    se3b = se3b / torch.norm(se3b)

    # 生成插值时间点 t
    t = torch.linspace(0, 1, new_time)

    # 计算插值所需的旋转角度
    thita = torch.arccos(torch.sum(se3a * se3b))

    # 使用 SLERP 公式计算新的 SE(3) 变换
    new_se3 = torch.stack(
        [(torch.sin((1 - time) * thita) * se3a + torch.sin(time * thita) * se3b) / torch.sin(thita) for time in t],
        dim=0)
    return new_se3


def load_poses(path):
    """
    Load SE(3) transformation matrices (4x4) from a file and return them as a list of tensors.
    从文件加载 SE(3) 变换矩阵（4x4），并将其返回为张量列表。
    """
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        # 将每一行数据转换为 4x4 的矩阵
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)

        # 反转 y 和 z 轴，以匹配预期的坐标系
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1

        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses


def interpolate_se3s(SE3s, num):
    """
    Perform SE(3) interpolation for a sequence of transformations.
    对一系列变换进行 SE(3) 插值。
    """
    li = Lie()
    # 将 SE(3) 转换为 SE(3) 向量表示
    se3s = li.SE3_to_se3(SE3s)
    interpolated_se3_poses = []

    for i in range(SE3s.shape[0] - 1):
        # 对每一对相邻的 SE(3) 变换进行插值
        if i != SE3s.shape[0] - 2:
            interpolated_se3_poses.append(interpolate_se3(se3s[i], se3s[i + 1], num)[:-1])
        else:
            interpolated_se3_poses.append(interpolate_se3(se3s[i], se3s[i + 1], num))

    # 将所有插值结果拼接成一个张量并返回
    return torch.cat(interpolated_se3_poses, dim=0)


def compute_velocity(se3_matrices, time_interval):
    """
    Compute velocity given a sequence of SE(3) matrices and a time interval.
    给定 SE(3) 变换矩阵序列和时间间隔，计算速度。
    """
    velocities = []
    for i in range(len(se3_matrices) - 1):
        # 计算相邻两帧的平移差异
        displacement = se3_matrices[i + 1][:3, 3] - se3_matrices[i][:3, 3]
        # 计算速度
        velocity = displacement / time_interval
        velocities.append(velocity)
    return torch.stack(velocities)


def compute_acceleration(velocities, time_interval):
    """
    Compute acceleration given a sequence of velocities and a time interval.
    给定速度序列和时间间隔，计算加速度。
    """
    accelerations = []
    for i in range(len(velocities) - 1):
        # 计算速度差异
        velocity_diff = velocities[i + 1] - velocities[i]
        # 计算加速度
        acceleration = velocity_diff / time_interval
        accelerations.append(acceleration)
    return torch.stack(accelerations)


def compute_angular_velocity(se3_matrices, time_interval):
    """
    Compute angular velocity given a sequence of SE(3) matrices and a time interval.
    给定 SE(3) 变换矩阵序列和时间间隔，计算角速度。
    """
    angular_velocities = []
    for i in range(len(se3_matrices) - 1):
        # 计算旋转矩阵的差异
        rotation_diff = se3_matrices[i + 1][:3, :3] @ se3_matrices[i][:3, :3].t()
        # 计算旋转差异的矩阵对数
        log_map = matrix_logarithm(rotation_diff)
        # 计算角速度
        angular_velocity = log_map / time_interval
        angular_velocities.append(angular_velocity)
    return torch.stack(angular_velocities)


def compute_angular_acceleration(angular_velocities, time_interval):
    """
    Compute angular acceleration given a sequence of angular velocities and a time interval.
    给定角速度序列和时间间隔，计算角加速度。
    """
    angular_accelerations = []
    for i in range(len(angular_velocities) - 1):
        # 计算角速度差异
        angular_velocity_diff = angular_velocities[i + 1] - angular_velocities[i]
        # 计算角加速度
        angular_acceleration = angular_velocity_diff / time_interval
        angular_accelerations.append(angular_acceleration)
    return torch.stack(angular_accelerations)


if __name__ == "__main__":
    # 在 t = 0.5 处进行插值
    t = 0.5
    lie_instance = Lie()
    # 加载并转换 SE(3) 矩阵
    poses = lie_instance.se3_to_matrix(torch.stack(load_poses("C:\\Users\\27215\\Desktop\\traj.txt"), dim=0))
    print(poses.shape)

    # 对 SE(3) 变换矩阵序列进行插值
    interpolated_poses = interpolate_se3s(poses, 7)
    print(interpolated_poses.shape)

    # 将插值后的 SE(3) 向量转换回 SE(3) 矩阵
    SE3_interpolated_poses = lie_instance.se3_to_SE3(interpolated_poses)

    # 计算速度、加速度、角速度和角加速度
    v = compute_velocity(SE3_interpolated_poses, 1 / 120)
    a = compute_acceleration(v, 1 / 120)
    w = compute_angular_velocity(SE3_interpolated_poses, 1 / 120)
    alpha = compute_angular_acceleration(w, 1 / 120)

    # 打印结果形状
    print(v.shape, a.shape)
    print(w.shape, alpha.shape)
