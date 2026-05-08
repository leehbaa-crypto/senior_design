"""
run_real_avoidance.py — SO-101 real robot hand/face avoidance using lerobot + placo IK.

Subscribes to ROS2 topics (/left_hand, /right_hand, /face_pose) from the
iPhone camera tracker, and controls the real SO-101 arm to avoid hands
and look toward faces.

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 run_real_avoidance.py --port /dev/ttyACM0
"""

import argparse
import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.model.kinematics import RobotKinematics

# ─── Avoidance parameters ─────────────────────────────────────────────────────
AVOIDANCE_THRESHOLD = 0.3
MAX_REPULSIVE_FORCE = 3.0
VIRTUAL_PADDING = 0.04
STEP_LIMIT_DEG = 2.0

# ─── Camera-to-world transform (same as sim) ──────────────────────────────────
CAMERA_DIRECTION = 1
CAMERA_OFFSET = np.array([0.25, -0.15, 0.50])
CAMERA_Y_SIGN = -1
CAMERA_Z_SIGN = -1


def transform_camera_to_world(cam_pos):
    raw_x = cam_pos[2] * CAMERA_DIRECTION
    raw_y = cam_pos[0] * CAMERA_Y_SIGN
    raw_z = cam_pos[1] * CAMERA_Z_SIGN
    return np.array([raw_x, raw_y, raw_z]) + CAMERA_OFFSET


# ─── ROS2 subscriber node ─────────────────────────────────────────────────────
class TrackingSubscriber(Node):
    def __init__(self):
        super().__init__('real_avoidance_node')
        self.create_subscription(Point, '/left_hand', self._cb('left'), 10)
        self.create_subscription(Point, '/right_hand', self._cb('right'), 10)
        self.create_subscription(Point, '/face_pose', self._cb('face'), 10)

        self.left_pose = None;  self.left_time = 0.0
        self.right_pose = None; self.right_time = 0.0
        self.face_pose = None;  self.face_time = 0.0

    def _cb(self, name):
        def callback(msg):
            if abs(msg.x) < 0.001 and abs(msg.y) < 0.001 and abs(msg.z) < 0.001:
                return
            pos = np.array([msg.x, msg.y, msg.z])
            now = time.time()
            if name == 'left':
                self.left_pose = pos; self.left_time = now
            elif name == 'right':
                self.right_pose = pos; self.right_time = now
            else:
                self.face_pose = pos; self.face_time = now
        return callback


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='/dev/ttyACM0')
    parser.add_argument('--id', type=str, default='so101')
    args = parser.parse_args()

    # ROS2 init
    rclpy.init()
    ros_node = TrackingSubscriber()
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    # Robot init
    config = SOFollowerRobotConfig(port=args.port, id=args.id, use_degrees=True)
    robot = SOFollower(config)
    robot.connect(calibrate=False)
    print(f"[Robot] Connected on {args.port}")

    # Kinematics init
    urdf_path = '/home/min/senior_design/urdf/so101.urdf'
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                   'wrist_flex', 'wrist_roll']
    kin = RobotKinematics(urdf_path, target_frame_name='gripper_frame_link',
                          joint_names=joint_names)

    # Read initial pose
    obs = robot.get_observation()
    q_deg = np.array([obs[f'{n}.pos'] for n in joint_names], dtype=float)
    print(f"[Robot] Initial joints (deg): {q_deg}")

    last_known_face = None
    anchor_q = q_deg.copy()

    print("[Robot] Avoidance loop started. Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            obs = robot.get_observation()
            q_deg = np.array([obs[f'{n}.pos'] for n in joint_names], dtype=float)

            # FK
            T_ee = kin.forward_kinematics(q_deg)
            ee_pos = T_ee[:3, 3]

            # Check active targets
            l_active = (now - ros_node.left_time) < 1.0 and ros_node.left_pose is not None
            r_active = (now - ros_node.right_time) < 1.0 and ros_node.right_pose is not None
            f_active = (now - ros_node.face_time) < 0.5 and ros_node.face_pose is not None

            if not l_active and not r_active and not f_active:
                time.sleep(0.02)
                continue

            # Collect hand world positions
            active_hands = []
            if l_active:
                active_hands.append(transform_camera_to_world(ros_node.left_pose))
            if r_active:
                active_hands.append(transform_camera_to_world(ros_node.right_pose))

            # Face tracking
            if f_active:
                raw_face = transform_camera_to_world(ros_node.face_pose)
                if last_known_face is None:
                    last_known_face = raw_face
                else:
                    last_known_face = 0.85 * last_known_face + 0.15 * raw_face

            # Repulsive force from hands on EE
            F_rep = np.zeros(3)
            is_dodging = False
            for hand_pos in active_hands:
                dist = np.linalg.norm(ee_pos - hand_pos)
                eff_dist = max(0.01, dist - VIRTUAL_PADDING)
                if eff_dist < AVOIDANCE_THRESHOLD:
                    is_dodging = True
                    danger = (AVOIDANCE_THRESHOLD - eff_dist) / AVOIDANCE_THRESHOLD
                    force_mag = MAX_REPULSIVE_FORCE * (danger ** 2)
                    direction = ee_pos - hand_pos
                    direction[2] = max(0.0, direction[2])
                    norm = np.linalg.norm(direction)
                    if norm > 0.001:
                        direction = direction / norm
                    F_rep += direction * force_mag

            # Attractive force toward anchor (rest position)
            attract_gain = 0.03 if is_dodging else 0.15
            pos_error = anchor_q - q_deg  # joint-space attraction to rest

            # Cartesian velocity command
            V_cart = np.zeros(6)
            V_cart[:3] = F_rep

            # If face detected, add orientation toward face
            target_look = last_known_face
            if target_look is not None:
                look_dir = target_look - ee_pos
                norm_look = np.linalg.norm(look_dir)
                if norm_look > 0.01:
                    look_dir = look_dir / norm_look
                    # Simple orientation: align EE x-axis toward face
                    # Use small orientation correction
                    R_curr = T_ee[:3, :3]
                    x_des = look_dir
                    up = np.array([0.0, 0.0, 1.0])
                    if abs(np.dot(x_des, up)) > 0.99:
                        up = np.array([0.0, 1.0, 0.0])
                    y_des = np.cross(up, x_des)
                    y_des = y_des / np.linalg.norm(y_des)
                    z_des = np.cross(x_des, y_des)
                    R_des = np.column_stack((x_des, y_des, z_des))

                    err_x = np.cross(R_curr[:, 0], R_des[:, 0])
                    err_y = np.cross(R_curr[:, 1], R_des[:, 1])
                    err_z = np.cross(R_curr[:, 2], R_des[:, 2])
                    e_ori = (err_x + err_y + err_z) / 2.0
                    if np.linalg.norm(e_ori) < 0.03:
                        e_ori = np.zeros(3)
                    V_cart[3:6] = 1.0 * e_ori

            # Jacobian pseudo-inverse → joint velocity
            # Use placo numerical Jacobian
            jac = np.zeros((6, len(joint_names)))
            delta = 0.5  # degrees
            T_base = kin.forward_kinematics(q_deg)
            pos_base = T_base[:3, 3]
            R_base = T_base[:3, :3]

            for i in range(len(joint_names)):
                q_pert = q_deg.copy()
                q_pert[i] += delta
                T_pert = kin.forward_kinematics(q_pert)
                jac[:3, i] = (T_pert[:3, 3] - pos_base) / np.deg2rad(delta)
                # Numerical rotation Jacobian (approximate)
                R_pert = T_pert[:3, :3]
                dR = R_pert @ R_base.T
                # Rodrigues → angular velocity
                angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
                if angle > 1e-6:
                    axis = np.array([dR[2, 1] - dR[1, 2],
                                     dR[0, 2] - dR[2, 0],
                                     dR[1, 0] - dR[0, 1]]) / (2 * np.sin(angle))
                    jac[3:6, i] = axis * angle / np.deg2rad(delta)

            # Damped pseudo-inverse
            lam = 0.1
            JtJ = jac.T @ jac + lam * np.eye(len(joint_names))
            J_pinv = np.linalg.inv(JtJ) @ jac.T

            dq_cart = J_pinv @ V_cart  # radians

            # Convert to degrees and add joint-space attraction
            dq_deg = np.rad2deg(dq_cart) + attract_gain * pos_error

            # Clip step size
            dq_deg = np.clip(dq_deg, -STEP_LIMIT_DEG, STEP_LIMIT_DEG)

            # Apply
            q_target = q_deg + dq_deg

            # Send to robot
            action = {f'{n}.pos': float(q_target[i]) for i, n in enumerate(joint_names)}
            action['gripper.pos'] = obs.get('gripper.pos', 50.0)
            robot.send_action(action)

            time.sleep(0.02)  # ~50 Hz control loop

    except KeyboardInterrupt:
        print("\n[Robot] Stopping...")
    finally:
        robot.disconnect()
        rclpy.shutdown()
        spin_thread.join()
        print("[Robot] Disconnected.")


if __name__ == '__main__':
    main()
