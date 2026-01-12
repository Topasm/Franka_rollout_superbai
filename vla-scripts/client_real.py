import time
from pathlib import Path
from typing import Dict, Any

import cv2
import json_numpy
import numpy as np
import pyrealsense2 as rs
import requests
from scipy.spatial.transform import Rotation as R
import pylibfranka

json_numpy.patch()

# Static configuration
SERVER_URL = "http://0.0.0.0:8000/act"
INSTRUCTION = "Pick up ball"
UNNORM_KEY = "austin_buds_dataset_converted_externally_to_rlds"
ROBOT_HOST = "172.16.0.2"
CAMERA_WIDTH = 400
CAMERA_HEIGHT = 400
REQUEST_TIMEOUT = 2.5
ENABLE_HOMING = True  # Set to False to disable homing on startup

# Safety limits per action
MAX_DPOS = 0.05  # meters - maximum position delta per action
MAX_DROT_RAD = 0.1  # radians per axis - maximum rotation delta per action

# Home joint position
HOME_JOINT_POSE = [-0.01588696, -0.25534376, 0.18628714, 
                   -2.28398158, 0.0769999, 2.02505396, 0.07858208]


def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def get_resized_frame(pipeline, width: int, height: int):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    return cv2.resize(color_image, (width, height))


def get_current_pose(robot):
    state = robot.read_once()
    O_T_EE = np.array(state.O_T_EE).reshape(4, 4).T
    translation = O_T_EE[:3, 3]
    rotation_matrix = O_T_EE[:3, :3]
    return translation, R.from_matrix(rotation_matrix).as_quat()


def move_to_pose(robot, translation, orientation_quat, steps: int = 50):
    target_T = np.eye(4)
    target_T[:3, 3] = translation
    target_T[:3, :3] = R.from_quat(orientation_quat).as_matrix()
    
    # Set collision behavior
    lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    robot.set_collision_behavior(
        lower_torque_thresholds,
        upper_torque_thresholds,
        lower_force_thresholds,
        upper_force_thresholds,
    )

    control = robot.start_cartesian_pose_control(
        pylibfranka.ControllerMode.CartesianImpedance
    )
    
    initial_T = None
    final_state = None
    time_elapsed = 0.0
    total_time = steps * 0.001
    
    for i in range(steps):
        state, duration = control.readOnce()
        time_elapsed += duration.to_sec()
        
        # Capture initial transformation on first iteration
        if initial_T is None:
            # Use O_T_EE_d (desired pose) or O_T_EE (measured pose)
            initial_T_array = state.O_T_EE_d if hasattr(state, 'O_T_EE_d') else state.O_T_EE
            initial_T = np.array(initial_T_array).reshape(4, 4).T
        
        final_state = state
        
        # Cubic interpolation with smooth acceleration (ease-in-out)
        # s(t) = 3t^2 - 2t^3 for t in [0,1]
        t = min(time_elapsed / total_time, 1.0)  # Normalized time [0,1]
        alpha = 3 * t**2 - 2 * t**3  # Smooth cubic S-curve
        
        # Interpolate transformation matrix
        # Position interpolation
        initial_pos = initial_T[:3, 3]
        target_pos = target_T[:3, 3]
        interp_pos = initial_pos + alpha * (target_pos - initial_pos)
        
        # Orientation interpolation using quaternion SLERP
        initial_rot = R.from_matrix(initial_T[:3, :3])
        target_rot = R.from_matrix(target_T[:3, :3])
        
        # Proper SLERP using scipy
        from scipy.spatial.transform import Slerp
        key_times = [0, 1]
        key_rots = R.from_quat([initial_rot.as_quat(), target_rot.as_quat()])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp(alpha)
        
        # Build interpolated transformation matrix
        interp_T = np.eye(4)
        interp_T[:3, 3] = interp_pos
        interp_T[:3, :3] = interp_rot.as_matrix()
        
        # Convert to column-major format for libfranka
        interp_T_col_major = interp_T.T.flatten()
        
        command = pylibfranka.CartesianPose(interp_T_col_major)
        
        # Set motion_finished when time elapsed exceeds total time
        if time_elapsed >= total_time:
            command.motion_finished = True
            
        control.writeOnce(command)
    
    return final_state


def move_to_joint_position(robot, target_q, steps: int = 100):
    """Move robot to target joint position with smooth interpolation."""
    target_q = np.array(target_q)
    
    # Set collision behavior
    lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
    lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
    robot.set_collision_behavior(
        lower_torque_thresholds,
        upper_torque_thresholds,
        lower_force_thresholds,
        upper_force_thresholds,
    )
    
    control = robot.start_joint_position_control(
        pylibfranka.ControllerMode.CartesianImpedance
    )
    
    initial_position = [0.0] * 7
    time_elapsed = 0.0
    
    # Control loop following official pylibfranka example
    for i in range(steps):
        state, duration = control.readOnce()
        
        # Update time
        time_elapsed += duration.to_sec()
        
        # On first iteration, capture initial position
        if time_elapsed <= duration.to_sec():
            initial_position = list(state.q_d if hasattr(state, 'q_d') else state.q)
        
        # Calculate smooth trajectory using cubic interpolation
        total_time = steps * 0.001  # Approximate total time
        t = min(time_elapsed / total_time, 1.0)
        delta = 3 * t**2 - 2 * t**3  # Cubic ease-in-out
        
        # Interpolate positions
        new_positions = []
        for j in range(7):
            new_positions.append(initial_position[j] + delta * (target_q[j] - initial_position[j]))
        
        # Set joint positions
        joint_positions = pylibfranka.JointPositions(new_positions)
        
        # Set motion_finished flag on last iteration
        if i >= steps - 1:
            joint_positions.motion_finished = True
        
        control.writeOnce(joint_positions)


def init_robot(robot, gripper):
    """Initialize robot to home position."""
    print("[INFO] Moving to home position...")
    try:
        move_to_joint_position(robot, HOME_JOINT_POSE, steps=2500)  # 2.5 seconds at 1kHz
        print("[INFO] Opening gripper...")
        gripper.move(0.08, 0.1)  # width, speed as positional args
        print("[INFO] Homing complete")
    except Exception as e:
        print(f"[WARNING] Homing failed: {e}")
        print("[INFO] Skipping homing, continuing with current position...")
        robot.automatic_error_recovery()


def clamp_action(delta: np.ndarray, limit: float):
    norm = np.linalg.norm(delta, ord=np.inf)
    if norm > limit:
        return delta * (limit / norm)
    return delta


def apply_action(robot, gripper, current_translation, current_rotation, action: Dict[str, Any]):
    # Convert numpy array to dictionary format if needed
    if isinstance(action, np.ndarray):
        if action.size == 7:
            action = {
                "dpos_x": action[0],
                "dpos_y": action[1],
                "dpos_z": action[2],
                "drot_x": action[3],
                "drot_y": action[4],
                "drot_z": action[5],
                "grip_command": "open" if action[6] < 0.5 else "close"
            }
        else:
            return current_translation, current_rotation
    
    if action.get("error"):
        return current_translation, current_rotation

    # Clamp deltas to safety limits
    dpos = np.array([
        action.get("dpos_x", 0.0),
        action.get("dpos_y", 0.0),
        action.get("dpos_z", 0.0),
    ], dtype=float)
    dpos = clamp_action(dpos, MAX_DPOS)

    drot = np.array([
        action.get("drot_x", 0.0),
        action.get("drot_y", 0.0),
        action.get("drot_z", 0.0),
    ], dtype=float)
    drot = clamp_action(drot, MAX_DROT_RAD)

    new_translation = current_translation + dpos
    rotation_increment = R.from_euler("xyz", drot)
    new_rotation = (R.from_quat(current_rotation) * rotation_increment).as_quat()

    try:
        final_state = move_to_pose(robot, new_translation, new_rotation, steps=1000)  # 1.0 second at 1kHz
        # Extract final pose from state returned by control loop
        if final_state:
            O_T_EE = np.array(final_state.O_T_EE).reshape(4, 4).T
            new_translation = O_T_EE[:3, 3]
            rotation_matrix = O_T_EE[:3, :3]
            new_rotation = R.from_matrix(rotation_matrix).as_quat()
        # Small settling delay to ensure robot is fully stopped
        time.sleep(0.1)
    except Exception as e:
        print(f"[WARNING] Move failed: {e}")
        try:
            robot.automatic_error_recovery()
            time.sleep(0.3)  # Longer recovery time
        except Exception:
            pass
        # Use current_translation/rotation (where we actually are) instead of target
        new_translation = current_translation
        new_rotation = current_rotation

    grip_command = action.get("grip_command")
    if grip_command:
        try:
            if grip_command == "close":
                gripper_state = gripper.read_once()
                if not gripper_state.is_grasped:
                    gripper.grasp(0.002, 0.2, 10, 0.04, 0.04)
            elif grip_command == "open":
                gripper.move(0.8, 0.2)
        except Exception:
            try:
                robot.automatic_error_recovery()
            except Exception:
                pass

    return new_translation, new_rotation


def main():
    print(f"[INFO] Connecting to robot at {ROBOT_HOST}...")
    robot = pylibfranka.Robot(ROBOT_HOST, pylibfranka.RealtimeConfig.kIgnore)
    gripper = pylibfranka.Gripper(ROBOT_HOST)
    robot.automatic_error_recovery()
    print("[INFO] Robot connected successfully")

    if ENABLE_HOMING:
        init_robot(robot, gripper)
    
    print("[INFO] Initializing camera...")
    pipeline = initialize_camera()
    print("[INFO] Camera initialized")
    
    current_translation, current_rotation = get_current_pose(robot)
    print(f"[INFO] Initial pose - Position: {current_translation}")
    print(f"[INFO] Server URL: {SERVER_URL}")
    print(f"[INFO] Instruction: {INSTRUCTION}")
    print(f"[INFO] Starting control loop...")

    try:
        while True:
            image = get_resized_frame(pipeline, CAMERA_WIDTH, CAMERA_HEIGHT)
            if image is None:
                continue

            payload = {
                "image": image,
                "instruction": INSTRUCTION,
                "unnorm_key": UNNORM_KEY,
            }

            try:
                response = requests.post(
                    SERVER_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                action = response.json()
                print(f"[ACTION] {action}")
            except Exception as e:
                print(f"[ERROR] Server request failed: {e}")
                continue

            try:
                current_translation, current_rotation = apply_action(
                    robot, gripper, current_translation, current_rotation, action
                )
            except Exception as e:
                print(f"[ERROR] Action execution failed: {e}")
                break

    finally:
        try:
            pipeline.stop()
            print("[INFO] Pipeline stopped")
        except Exception:
            pass


if __name__ == "__main__":
    main()
