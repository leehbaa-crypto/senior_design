"""
run_real_avoidance.py — SO-101 face tracking (simple mode).

Motors 2/3/4 locked. Motor 1 tracks nose to image center.
Motor 5 aligns eye-line to horizontal (requires /face_roll topic).

Usage:
    source /opt/ros/humble/setup.bash
    python3 run_real_avoidance.py --port /dev/ttyACM0
"""

import argparse
import math
import select
import sys
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


# ─── Tracking gains ───────────────────────────────────────────────────────────
PAN_KP = 30.0           # shoulder_pan P gain (deg per radian of lateral offset)
PAN_KI = 50.0            # shoulder_pan I gain (deg per radian·s)
ROLL_KP = 30.0          # wrist_roll P gain (deg per radian of eye tilt)
ROLL_KI = 3.0            # wrist_roll I gain (deg per radian·s)
I_CLAMP = 20.0          # integral windup limit (deg)
STEP_LIMIT = 20.0      # max deg change per cycle


# ─── ROS2 subscriber ─────────────────────────────────────────────────────────
class TrackingSubscriber(Node):
    def __init__(self):
        super().__init__('simple_face_track')
        self.create_subscription(Point, '/face_pose', self._cb, 10)
        self.create_subscription(Float32, '/face_roll', self._roll_cb, 10)
        self.face_pos = None
        self.face_time = 0.0
        self.face_roll = None
        self.roll_time = 0.0

    def _cb(self, msg):
        if abs(msg.x) < 0.001 and abs(msg.y) < 0.001 and abs(msg.z) < 0.001:
            return
        self.face_pos = np.array([msg.x, msg.y, msg.z])
        self.face_time = time.time()

    def _roll_cb(self, msg):
        self.face_roll = msg.data
        self.roll_time = time.time()


# ─── Debug panel ──────────────────────────────────────────────────────────────
def draw_debug(q, names, cam, pan_err, roll_err, active, estop):
    h, w = 400, 350
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y = 25
    def t(s, c=(200, 200, 200)):
        nonlocal y
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1); y += 20

    t(f"{'E-STOP' if estop else 'RUNNING'}", (0,0,255) if estop else (0,255,0))
    t(f"Face: {'ON' if active else 'OFF'}")
    y += 8
    t("-- Joints (deg) --", (255,200,0))
    for i, n in enumerate(names):
        t(f"  {n:15s}: {q[i]:7.1f}")
    y += 8
    t("-- Tracking --", (255,200,0))
    if cam is not None:
        t(f"  Cam x={cam[0]:+.3f} y={cam[1]:+.3f} z={cam[2]:.3f}")
    t(f"  Pan err:  {pan_err:+.1f} deg" if pan_err is not None else "  Pan err:  ---")
    t(f"  Roll err: {roll_err:+.1f} deg" if roll_err is not None else "  Roll err: ---")

    # crosshair
    cx, cy = w // 2, h - 60
    cv2.line(img, (cx-25, cy), (cx+25, cy), (80,80,80), 1)
    cv2.line(img, (cx, cy-25), (cx, cy+25), (80,80,80), 1)
    if cam is not None:
        fx = int(np.clip(cx + cam[0] * 400, 10, w-10))
        fy = int(np.clip(cy - cam[1] * 400, 10, h-10))
        cv2.circle(img, (fx, fy), 7, (0,255,255), -1)

    cv2.imshow("Debug", img)
    cv2.waitKey(1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--id', default='face_tracker')
    args = parser.parse_args()

    # ROS2
    rclpy.init()
    ros = TrackingSubscriber()
    threading.Thread(target=rclpy.spin, args=(ros,), daemon=True).start()

    # Robot
    config = SOFollowerRobotConfig(port=args.port, id=args.id, use_degrees=True)
    robot = SOFollower(config)
    robot.connect(calibrate=True)
    robot.bus.disable_torque("gripper")

    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                   'wrist_flex', 'wrist_roll']

    obs = robot.get_observation()
    anchor = np.array([obs[f'{n}.pos'] for n in joint_names], dtype=float)
    print(f"[Robot] Anchor: {anchor}")

    # Locked joints (2,3,4) — always hold anchor values
    LOCKED_IDX = {1, 2, 3}

    estop = False
    def kb():
        nonlocal estop
        while not estop:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if sys.stdin.read(1) == ' ':
                    estop = True; print("\n[E-STOP]")
    threading.Thread(target=kb, daemon=True).start()

    print("[Robot] Tracking. SPACE=e-stop, Ctrl+C=quit.")
    cam_smooth = None
    pan_err_deg = None
    roll_err_deg = None
    i_pan = 0.0           # pan integral accumulator (rad·s)
    i_roll = 0.0          # roll integral accumulator (rad·s)
    t_prev = time.time()

    try:
        while not estop:
            now = time.time()
            dt = now - t_prev
            t_prev = now
            obs = robot.get_observation()
            q = np.array([obs[f'{n}.pos'] for n in joint_names], dtype=float)

            active = (now - ros.face_time) < 0.5 and ros.face_pos is not None

            target = anchor.copy()

            if active:
                cam = ros.face_pos.copy()
                # smooth
                cam_smooth = cam if cam_smooth is None else 0.7 * cam_smooth + 0.3 * cam

                # Motor 1 (shoulder_pan): center nose horizontally
                # cam_smooth[0] = lateral offset (+ = right in camera)
                pan_err_rad = math.atan2(cam_smooth[0], cam_smooth[2])
                pan_err_deg = math.degrees(pan_err_rad)
                i_pan = np.clip(i_pan + pan_err_rad * dt, -I_CLAMP, I_CLAMP)
                dq_pan = PAN_KP * pan_err_rad + PAN_KI * i_pan  # PI control
                target[0] = anchor[0] - dq_pan  # negate to correct

                # Motor 5 (wrist_roll): align eye-line horizontal
                roll_active = (now - ros.roll_time) < 0.5 and ros.face_roll is not None
                if roll_active:
                    roll_err_rad = ros.face_roll  # eye-line tilt from camera
                    roll_err_deg = math.degrees(roll_err_rad)
                    i_roll = np.clip(i_roll + roll_err_rad * dt, -I_CLAMP, I_CLAMP)
                    dq_roll = ROLL_KP * roll_err_rad + ROLL_KI * i_roll  # PI control
                    target[4] = anchor[4] + dq_roll
                else:
                    roll_err_deg = None
                    i_roll = 0.0

            else:
                cam_smooth = None
                pan_err_deg = None
                roll_err_deg = None
                i_pan = 0.0
                i_roll = 0.0

            # Apply locked joints
            for idx in LOCKED_IDX:
                target[idx] = anchor[idx]

            # Clip step size
            dq = np.clip(target - q, -STEP_LIMIT, STEP_LIMIT)
            q_out = q + dq

            action = {f'{n}.pos': float(q_out[i]) for i, n in enumerate(joint_names)}
            action['gripper.pos'] = obs.get('gripper.pos', 50.0)
            robot.send_action(action)

            draw_debug(q, joint_names, cam_smooth, pan_err_deg, roll_err_deg, active, estop)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[Robot] Stopping...")
    finally:
        if estop:
            robot.bus.disable_torque()
        robot.disconnect()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("[Robot] Disconnected.")


if __name__ == '__main__':
    main()
