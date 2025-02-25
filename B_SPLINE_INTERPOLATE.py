import torch
import numpy as np
from scipy.interpolate import BSpline
from rotation_conversions import Lie
import torch.nn.functional as F


def matrix_logarithm(R):
    """
    Compute the matrix logarithm of a 3x3 rotation matrix R.
    This function computes the angular velocity (as a vector) corresponding to the
    difference between two 3D rotation matrices using the matrix logarithm.
    """
    # Ensure R is a rotation matrix (normalize)
    _R = F.normalize(R, dim=(0, 1))

    # Calculate the angle of rotation using the trace of the matrix
    angle = torch.acos((torch.trace(_R) - 1) / 2)

    if angle < 1e-5:
        # For near identity rotations (small angle), use first-order approximation
        omega = torch.stack([_R[2, 1] - _R[1, 2], _R[0, 2] - _R[2, 0], _R[1, 0] - _R[0, 1]]) / 2
    else:
        # General case for non-small angles, use the logarithmic map formula
        omega = angle / (2 * torch.sin(angle)) * torch.stack([
            _R[2, 1] - _R[1, 2],
            _R[0, 2] - _R[2, 0],
            _R[1, 0] - _R[0, 1]
        ])

    return omega


def compute_velocity(se3_matrices, time_interval):
    """
    Compute the linear velocity of a rigid body from its SE(3) transformation matrices.
    The velocity is computed as the displacement between consecutive poses divided by the time interval.
    """
    velocities = []
    for i in range(len(se3_matrices) - 1):
        # Compute the displacement between consecutive poses (translations only)
        displacement = se3_matrices[i + 1][:3, 3] - se3_matrices[i][:3, 3]
        # Calculate velocity by dividing displacement by time interval
        velocity = displacement / time_interval
        velocities.append(velocity)
    return torch.stack(velocities)


def compute_acceleration(velocities, time_interval):
    """
    Compute the linear acceleration of a rigid body from its velocities.
    The acceleration is computed as the difference between consecutive velocities divided by the time interval.
    """
    accelerations = []
    for i in range(len(velocities) - 1):
        # Calculate the difference in velocity (acceleration)
        velocity_diff = velocities[i + 1] - velocities[i]
        acceleration = velocity_diff / time_interval
        accelerations.append(acceleration)
    return torch.stack(accelerations)


def compute_angular_velocity(se3_matrices, time_interval):
    """
    Compute the angular velocity of a rigid body from its SE(3) transformation matrices.
    The angular velocity is computed using the matrix logarithm of the rotation matrix difference.
    """
    angular_velocities = []
    for i in range(len(se3_matrices) - 1):
        # Compute the relative rotation between consecutive poses
        rotation_diff = se3_matrices[i + 1][:3, :3] @ se3_matrices[i][:3, :3].t()
        # Compute the angular velocity using the matrix logarithm of the relative rotation
        log_map = matrix_logarithm(rotation_diff)
        angular_velocity = log_map / time_interval
        angular_velocities.append(angular_velocity)
    return torch.stack(angular_velocities)


def compute_angular_acceleration(angular_velocities, time_interval):
    """
    Compute the angular acceleration of a rigid body from its angular velocities.
    The angular acceleration is computed as the difference between consecutive angular velocities divided by the time interval.
    """
    angular_accelerations = []
    for i in range(len(angular_velocities) - 1):
        # Calculate the difference in angular velocity (angular acceleration)
        angular_velocity_diff = angular_velocities[i + 1] - angular_velocities[i]
        angular_acceleration = angular_velocity_diff / time_interval
        angular_accelerations.append(angular_acceleration)
    return torch.stack(angular_accelerations)


def load_poses(path):
    """
    Load SE(3) transformation matrices (4x4) from a text file.
    Each line in the file represents a transformation matrix, which is read and converted into a tensor.
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
    # Load 3D trajectory data consisting of SE(3) transformation matrices
    data = torch.stack(load_poses("C:\\Users\\27215\\Desktop\\traj.txt"), dim=0)
    print(data.shape)

    # Extract rotation and translation components from the SE(3) matrices
    rotation_data = data[:, :3, :3]
    translation_data = data[:, :3, 3]

    # Convert PyTorch Tensors to NumPy arrays for B-spline interpolation
    rotation_data_np = rotation_data.numpy()
    translation_data_np = translation_data.numpy()

    # Create B-spline interpolation objects for rotation and translation
    degree = 3  # Degree of B-spline (cubic interpolation)
    t = np.arange(len(rotation_data_np))
    spline_rotation = BSpline(t, rotation_data_np, degree)
    spline_translation = BSpline(t, translation_data_np, degree)

    # Define new, denser time points for interpolation
    original_time_points = np.arange(len(data))
    new_time_points = np.linspace(0, len(data) - 1, 10 * (len(data) - 1) + 1)

    # Perform the interpolation for rotation and translation
    interpolated_rotation = torch.from_numpy(spline_rotation(new_time_points)).float()
    interpolated_translation = torch.from_numpy(spline_translation(new_time_points)).float()

    # Combine the interpolated rotation and translation to form the SE(3) matrices
    interpolated_se3 = torch.cat((interpolated_rotation, interpolated_translation.unsqueeze(2)), dim=2)

    # Compute velocity, acceleration, and angular velocity
    v = compute_velocity(interpolated_se3, 1 / 200)
    a = compute_acceleration(v, 1 / 200)  # Acceleration
    w = compute_angular_velocity(interpolated_se3, 1 / 200)  # Angular velocity

    print(interpolated_se3.shape)
    print(a.shape)
    print(w.shape)

    dt = 1 / 200  # Time step

    # Integrate acceleration to get velocity using cumulative sum
    velocity = torch.cat([torch.tensor([[0, 0, 0]]), torch.cumsum(a, dim=0) * dt]) + v[0]

    # Integrate velocity to get displacement using cumulative sum
    displacement = torch.cat([torch.tensor([[0, 0, 0]]) + torch.cumsum(velocity, dim=0) * dt]) + interpolated_se3[0][:3, 3]

    print(displacement[0], interpolated_se3[0][:3, 3])
    print(displacement.shape, interpolated_se3.shape, translation_data.shape)

    print(translation_data[-1])
    print(interpolated_se3[-1])

    print(displacement[-1])
