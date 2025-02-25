import torch
import numpy as np
from scipy.interpolate import CubicSpline
from rotation_conversions import Lie
from B_SPLINE_INTERPOLATE import *


def load_poses(path):
    """
    Load SE(3) transformation matrices (4x4) from a text file.
    Each line in the file represents a transformation matrix, which is read and converted into a tensor.
    The coordinate system is modified by flipping the y and z axes to match the expected conventions.
    """
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        # Convert each line into a 4x4 transformation matrix
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        # Modify coordinate system to match expected conventions (inverting axes)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses

if __name__ == "__main__":
    lie_instance = Lie()
    # 假设 data 是包含 2000 个 SE(3) 变换矩阵的 3D Tensor
    # Load 3D trajectory data consisting of SE(3) transformation matrices
    data = torch.stack(load_poses("C:\\Users\\27215\\Desktop\\traj.txt"), dim=0)
    print(data.shape)

    # 提取旋转和平移部分
    # Extract the rotation (3x3) and translation (3x1) parts from the SE(3) matrices
    rotation_data = data[:, :3, :3]
    translation_data = data[:, :3, 3]

    # 将 PyTorch Tensor 转换为 NumPy 数组
    # Convert PyTorch Tensors to NumPy arrays for cubic spline interpolation
    rotation_data_np = rotation_data.numpy()
    translation_data_np = translation_data.numpy()

    # 创建样条插值对象
    # Create cubic spline interpolation objects for rotation and translation
    cubic_spline_rotation = CubicSpline(np.arange(len(rotation_data_np)), rotation_data_np, axis=0)
    cubic_spline_translation = CubicSpline(np.arange(len(translation_data_np)), translation_data_np, axis=0)

    # 生成更密集的时间点
    # Define new, denser time points for interpolation
    original_time_points = np.arange(len(data))
    new_time_points = np.linspace(0, len(data) - 1, 6 * (len(data) - 1) + 1)

    # 进行插值
    # Perform cubic spline interpolation for both rotation and translation
    interpolated_rotation = torch.from_numpy(cubic_spline_rotation(new_time_points)).float()
    interpolated_translation = torch.from_numpy(cubic_spline_translation(new_time_points)).float()

    # 组合插值结果
    # Combine the interpolated rotation and translation to form the SE(3) matrices
    interpolated_se3 = torch.cat((interpolated_rotation, interpolated_translation.unsqueeze(2)), dim=2)


    v = compute_velocity(interpolated_se3, 1 / 200)
    a = compute_acceleration(v, 1 / 200)  # Acceleration
    w = compute_angular_velocity(interpolated_se3, 1 / 200)  # Angular velocity
    # alpha = compute_angular_acceleration(w, 1 / 120)
    dt = 1 / 200  # Time step

    # 第一步：加速度积分到速度
    # First step: Integrate acceleration to obtain velocity using cumulative sum
    velocity = torch.cumsum(a, dim=0) * dt + v[0]

    # 第二步：速度积分到位移
    # Second step: Integrate velocity to obtain displacement using cumulative sum
    displacement = torch.cumsum(velocity, dim=0) * dt + translation_data[0]

    # 第三步：位移积分到位姿变化
    # Third step: Integrate displacement to obtain pose change using cumulative sum
    pose_change = torch.cumsum(displacement, dim=0) * dt

    # Print the results for verification
    print(translation_data[-1])
    print(interpolated_se3[-1])

    print(displacement[-1] - displacement[0])
    print(displacement[-1])
