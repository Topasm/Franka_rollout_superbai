import json
import time
import numpy as np
import logging
import cv2
import traceback
from pathlib import Path
from typing import Any, Dict, Union
import torch
import json_numpy
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transformers import AutoModelForVision2Seq, AutoProcessor
import pylibfranka
import pyrealsense2 as rs
import os

json_numpy.patch()

# Constants
MOVE_INCREMENT = 0.0002
SPEED = 0.05
FORCE = 20.0
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Global variables to manage image files
# image_dir = '/home/ahrilab/Desktop/openvla/d'
# image_files = sorted([os.path.join(image_dir, f)
#                      for f in os.listdir(image_dir) if f.endswith('.jpg')])
# image_index = 0


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    """Constructs the prompt based on the openvla version."""
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class OpenVLAController:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: str = "flash_attention_2"):
        self.openvla_path = openvla_path
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

    def predict_action(self, image: np.ndarray, instruction: str) -> Dict[str, Any]:
        """Predicts an action based on the provided image and instruction."""
        try:
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert(
                "RGB")).to(self.device, dtype=torch.bfloat16)
            action_output = self.vla.predict_action(
                **inputs, unnorm_key="custom_wow_dataset", do_sample=False)
            print(action_output)

            # Ensure action_output is parsed correctly into a dictionary format
            if isinstance(action_output, torch.Tensor):
                # convert to numpy if still in tensor format
                action_output = action_output.cpu().numpy()

            # Convert numpy array to dictionary if it has predefined structure
            if isinstance(action_output, np.ndarray) and action_output.size == 7:
                return {
                    "dpos_x": action_output[0],
                    "dpos_y": action_output[1],
                    "dpos_z": action_output[2],
                    "drot_x": action_output[3],
                    "drot_y": action_output[4],
                    "drot_z": action_output[5],
                    "grip_command": "open" if action_output[6] < 0.5 else "close"
                }

            return action_output  # assuming action_output is already in dictionary format
        except Exception as e:
            logging.error(traceback.format_exc())
            return {"error": str(e)}


def initialize_camera():
    """Initializes and starts the camera pipeline with custom settings."""
    pipeline = rs.pipeline()
    config = rs.config()
    # Use a supported resolution
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Get the device and set exposure, gain, and white balance
    device = profile.get_device()
    sensor = device.query_sensors()[1]  # Color sensor is usually at index 1
    # sensor.set_option(rs.option.exposure, 120)        # Set exposure
    # sensor.set_option(rs.option.gain, 0)              # Set gain
    # sensor.set_option(rs.option.white_balance, 3900)  # Set white balance

    return pipeline, profile


def get_resized_frame(pipeline, width=400, height=400):
    """Captures a frame from the camera and resizes it to the specified dimensions."""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None

    color_image = np.asanyarray(color_frame.get_data())
    resized_image = cv2.resize(color_image, (width, height))
    return resized_image


def get_current_pose(robot):
    """Get current end-effector position and orientation from robot."""
    try:
        state = robot.read_once()
        # O_T_EE is column-major 16-element array, convert to 4x4 row-major
        O_T_EE = state.O_T_EE.reshape(4, 4).T
        translation = O_T_EE[:3, 3]
        rotation_matrix = O_T_EE[:3, :3]
        # Convert to quaternion
        orientation_quat = R.from_matrix(rotation_matrix).as_quat()
        return translation, orientation_quat
    except Exception as e:
        logging.error(f"Error reading robot pose: {e}")
        raise


def move_to_joint_position(robot, target_q, steps=100):
    """Move robot to target joint position using simple control loop."""
    try:
        target_q = np.array(target_q)
        control = robot.start_joint_position_control(pylibfranka.ControllerMode.JointImpedance)
        
        for i in range(steps):
            state, duration = control.readOnce()
            command = pylibfranka.JointPositions(target_q)
            control.writeOnce(command)
            time.sleep(0.001)  # Approximate 1 kHz
        
        return True
    except Exception as e:
        logging.error(f"Error moving to joint position: {e}")
        return False


def move_to_pose(robot, translation, orientation_quat, steps=50):
    """Move robot to target Cartesian pose using simple control loop."""
    try:
        # Build 4x4 transformation matrix
        target_T = np.eye(4)
        target_T[:3, 3] = translation
        target_T[:3, :3] = R.from_quat(orientation_quat).as_matrix()
        # Convert to column-major format (flatten transposed)
        target_T_col_major = target_T.T.flatten()
        
        control = robot.start_cartesian_pose_control(pylibfranka.ControllerMode.CartesianImpedance)
        
        for i in range(steps):
            state, duration = control.readOnce()
            command = pylibfranka.CartesianPose(target_T_col_major)
            control.writeOnce(command)
            time.sleep(0.001)  # Approximate 1 kHz
        
        return True
    except Exception as e:
        logging.error(f"Error moving to pose: {e}")
        return False


# def get_next_image(width=224, height=224):
#     """Reads the next image from the directory and resizes it."""
#     global image_index
#     if image_index >= len(image_files):
#         image_index = 0  # Loop back to the first image
#     image_path = image_files[image_index]
#     image_index += 1

#     image = cv2.imread(image_path)
#     if image is None:
#         return None
#     resized_image = cv2.resize(image, (width, height))

#     return resized_image


def init_robot(robot, gripper):
    """Initialize robot to home position."""
    joint_pose = [-0.01588696, -0.25534376, 0.18628714, -
                  2.28398158, 0.0769999, 2.02505396, 0.07858208]
    
    try:
        move_to_joint_position(robot, joint_pose)
        success = gripper.move(width=0.8, speed=0.1)
        if not success:
            logging.warning("Gripper move command failed")
    except Exception as e:
        logging.error(f"Error initializing robot: {e}")
        raise


def main():
    """Main function to initialize robot, camera, and run action prediction in a loop."""
    hostname = '172.16.0.2'
    
    # Initialize robot and gripper with pylibfranka
    try:
        robot = pylibfranka.Robot(hostname, pylibfranka.RealtimeConfig.kIgnore)
        gripper = pylibfranka.Gripper(hostname)
        robot.automatic_error_recovery()
        logging.info("Robot and gripper initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize robot: {e}")
        return
    
    # Optionally initialize robot to home position
    # init_robot(robot, gripper=gripper)

    # Get current pose
    try:
        current_translation, current_rotation = get_current_pose(robot)
    except Exception as e:
        logging.error(f"Failed to read initial pose: {e}")
        return
    
    pipeline, profile = initialize_camera()

    # Initialize OpenVLA Controller
    controller = OpenVLAController(
        openvla_path="/home/ahrilab/Desktop/sh_model2")

    while 1:  # Maintain loop at 1000 Hz
        image = get_resized_frame(pipeline=pipeline, width=400, height=400)
        # img_path = './d/image/place_image/'
        # img_list = os.listdir(img_path)
        # img_list.sort()
        # print(img_list)

        # for img_i in img_list:
        #     image = Image.open(img_path+img_i)
        #     image = np.array(image)

        if image is None:
            continue
        cv2.imshow("Model Input - Real-Time View", image)
        # Image._show(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the display
            break
        # Set instruction (customizable based on task)
        instruction = "Place the block on the plate."  # Example instruction

        # # Predict action
        action = controller.predict_action(image, instruction)

        # print(img_i)
        # time.sleep(2)

        if "error" not in action:
            try:
                # Update position and orientation based on action
                if 'dpos_x' in action and 'dpos_y' in action and 'dpos_z' in action:
                    current_translation += np.array([
                        action['dpos_x'],
                        action['dpos_y'],
                        action['dpos_z']
                    ])
                if 'drot_x' in action and 'drot_y' in action and 'drot_z' in action:
                    euler_angles = np.array([
                        action['drot_x'],
                        action['drot_y'],
                        action['drot_z']
                    ])
                    rotation_increment = R.from_euler(
                        'xyz', euler_angles).as_quat()
                    rotation_increment[:] = [0, 0, 0, 1]
                    current_rotation = R.from_quat(
                        current_rotation) * R.from_quat(rotation_increment)
                    current_rotation = current_rotation.as_quat()

                # Move the robot to the new pose
                success = move_to_pose(robot, current_translation, current_rotation)
                if not success:
                    logging.warning("Failed to execute pose movement")
                    # Try to recover
                    robot.automatic_error_recovery()
                    current_translation, current_rotation = get_current_pose(robot)

                # Control the gripper based on the grip_command
                if 'grip_command' in action:
                    try:
                        if action['grip_command'] == 'close':
                            gripper_state = gripper.read_once()
                            if not gripper_state.is_grasped:
                                success = gripper.grasp(0.002, 0.2, 10, 0.04, 0.04)
                                if not success:
                                    logging.warning("Grasp command failed")
                        elif action['grip_command'] == 'open':
                            success = gripper.move(0.8, 0.2)
                            if not success:
                                logging.warning("Gripper open command failed")
                    except Exception as e:
                        logging.error(f"Gripper control error: {e}")
                        
            except (pylibfranka.NetworkException, pylibfranka.ControlException, 
                    pylibfranka.CommandException) as e:
                logging.error(f"Robot control error: {e}")
                try:
                    robot.automatic_error_recovery()
                    current_translation, current_rotation = get_current_pose(robot)
                except Exception as recovery_error:
                    logging.error(f"Error recovery failed: {recovery_error}")
                    break
            except Exception as e:
                logging.error(f"Unexpected error during action execution: {e}")
                logging.error(traceback.format_exc())

        # print("end of ", img_i)
        # break


if __name__ == "__main__":
    main()
