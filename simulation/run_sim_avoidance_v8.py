import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import mujoco
import mujoco.viewer
import numpy as np
import threading
import os
import time

# ==========================================
# 🛠️ [설정 값] 🛠️
# ==========================================
CAMERA_DIRECTION = 1

# [카메라 위치 설정]
CAMERA_OFFSET = np.array([0.25, -0.15, 0.55]) 

# 🛠️ [수정됨] 회피 민감도 하향 조정 (덜 보수적으로)
AVOIDANCE_THRESHOLD = 0.35  # (기존 0.5) 35cm 안으로 들어와야 피하기 시작
MAX_REPULSIVE_FORCE = 2.5   # (기존 5.0) 밀어내는 힘을 반으로 줄여 부드럽게 피함
VIRTUAL_PADDING = 0.03      # (기존 0.1) 가상 보호막 두께를 3cm로 대폭 줄임

class MuJoCoROS2Node(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.create_subscription(Point, '/left_hand', self.left_cb, 10)
        self.create_subscription(Point, '/right_hand', self.right_cb, 10)
        self.create_subscription(Point, '/face_pose', self.face_cb, 10)
        
        self.left_pose = None; self.left_time = 0.0
        self.right_pose = None; self.right_time = 0.0
        self.face_pose = None; self.face_time = 0.0
        
    def left_cb(self, msg):
        self.left_pose = np.array([msg.x, msg.y, msg.z]); self.left_time = time.time()
    def right_cb(self, msg):
        self.right_pose = np.array([msg.x, msg.y, msg.z]); self.right_time = time.time()
    def face_cb(self, msg):
        self.face_pose = np.array([msg.x, msg.y, msg.z]); self.face_time = time.time()

def transform_camera_to_world(cam_pos):
    world_x = cam_pos[2] * CAMERA_DIRECTION
    world_y = -cam_pos[0] * CAMERA_DIRECTION 
    world_z = -cam_pos[1] 
    return np.array([world_x, world_y, world_z]) + CAMERA_OFFSET

def main(args=None):
    rclpy.init(args=args)
    ros_node = MuJoCoROS2Node()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(current_dir, "scene.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joint_ids = [model.joint(name).qposadr[0] for name in joint_names]
        
        ee_site_name = "gripperframe"
        try:
            ee_id = model.site(ee_site_name).id; is_site = True
        except KeyError:
            ee_id = model.body("gripper").id; is_site = False

        mujoco.mj_step(model, data)
        initial_q = np.array([data.qpos[idx] for idx in joint_ids])
        anchor_pos = data.site_xpos[ee_id].copy() if is_site else data.xpos[ee_id].copy()
        
        smoothed_q = initial_q.copy() 
        filtered_dq = np.zeros(len(joint_ids))
        
        last_render_time = time.time()
        last_time_filter = time.time()
        prev_face_world = None
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🚀 Active Node v11 Running: Relaxed Avoidance & Cylindrical Camera")
            
            while viewer.is_running():
                curr_time = time.time()
                dt = curr_time - last_time_filter
                if dt <= 0: dt = 0.001
                
                l_avoid_active = (curr_time - ros_node.left_time) < 1.5 and ros_node.left_pose is not None
                r_avoid_active = (curr_time - ros_node.right_time) < 1.5 and ros_node.right_pose is not None
                f_active = (curr_time - ros_node.face_time) < 0.5 and ros_node.face_pose is not None
                
                if not l_avoid_active and not r_avoid_active and not f_active:
                    for i, idx in enumerate(joint_ids):
                        target_dq = (initial_q[i] - smoothed_q[i]) * 0.03
                        filtered_dq[i] = 0.9 * filtered_dq[i] + 0.1 * target_dq
                        filtered_dq[i] = np.clip(filtered_dq[i], -0.05, 0.05)
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]
                    prev_face_world = None
                else:
                    ee_pos = data.site_xpos[ee_id] if is_site else data.xpos[ee_id]
                    curr_ori_mat = data.site_xmat[ee_id].reshape(3, 3) if is_site else data.xmat[ee_id].reshape(3, 3)
                    
                    active_hands = []
                    if l_avoid_active: active_hands.append(transform_camera_to_world(ros_node.left_pose))
                    if r_avoid_active: active_hands.append(transform_camera_to_world(ros_node.right_pose))

                    is_dodging = False
                    closest_hand_pos = None
                    min_global_dist = 999.0

                    F_repulsive_ee = np.zeros(3)
                    for hand_pos in active_hands:
                        dist_to_ee = np.linalg.norm(ee_pos - hand_pos)
                        if dist_to_ee < min_global_dist:
                            min_global_dist = dist_to_ee
                            closest_hand_pos = hand_pos
                            
                        eff_dist = max(0.01, dist_to_ee - VIRTUAL_PADDING)
                        if eff_dist < AVOIDANCE_THRESHOLD:
                            is_dodging = True
                            force_mag = MAX_REPULSIVE_FORCE * ((AVOIDANCE_THRESHOLD - eff_dist) / AVOIDANCE_THRESHOLD)**2
                            F_repulsive_ee += ((ee_pos - hand_pos) / dist_to_ee) * force_mag

                    target_look_pos = transform_camera_to_world(ros_node.face_pose) if f_active else closest_hand_pos
                    V_err = np.zeros(6) 
                    
                    F_attract = 0.5 * (anchor_pos - ee_pos) 
                    V_err[0:3] = F_attract + F_repulsive_ee
                    
                    if target_look_pos is not None:
                        x_des = target_look_pos - ee_pos
                        norm_x = np.linalg.norm(x_des)
                        if norm_x > 0.01:
                            x_des = x_des / norm_x
                            up = np.array([0.0, 0.0, 1.0]) 
                            if abs(np.dot(x_des, up)) > 0.99: up = np.array([0.0, 1.0, 0.0]) 
                            y_des = np.cross(up, x_des)
                            y_des = y_des / np.linalg.norm(y_des)
                            z_des = np.cross(x_des, y_des)
                            R_des = np.column_stack((x_des, y_des, z_des))
                            err_x = np.cross(curr_ori_mat[:, 0], R_des[:, 0])
                            err_y = np.cross(curr_ori_mat[:, 1], R_des[:, 1])
                            err_z = np.cross(curr_ori_mat[:, 2], R_des[:, 2])
                            V_err[3:6] = 1.5 * ((err_x + err_y + err_z) / 2.0)

                    jacp_ee = np.zeros((3, model.nv))
                    jacr_ee = np.zeros((3, model.nv))
                    if is_site: mujoco.mj_jacSite(model, data, jacp_ee, jacr_ee, ee_id)
                    else: mujoco.mj_jacBody(model, data, jacp_ee, jacr_ee, ee_id)
                    J_ee = np.vstack((jacp_ee, jacr_ee))
                    
                    lambda_sq = 0.01
                    J_pinv = J_ee.T @ np.linalg.inv(J_ee @ J_ee.T + lambda_sq * np.eye(6))
                    
                    dq_look = J_pinv @ V_err
                    
                    dq_posture = np.zeros(model.nv)
                    if f_active and not is_dodging:
                        bias_shoulder = -1.0 
                        bias_elbow = -1.5    
                        bias_wrist = 1.0     
                        
                        idx_shoulder = model.joint("shoulder_lift").dofadr[0]
                        idx_elbow = model.joint("elbow_flex").dofadr[0]
                        idx_wrist = model.joint("wrist_flex").dofadr[0]
                        
                        dq_posture[idx_shoulder] = (initial_q[1] + bias_shoulder - smoothed_q[1]) * 0.5
                        dq_posture[idx_elbow] = (initial_q[2] + bias_elbow - smoothed_q[2]) * 0.5
                        dq_posture[idx_wrist] = (initial_q[3] + bias_wrist - smoothed_q[3]) * 0.5

                    total_dq = dq_look + dq_posture

                    alpha = 0.9 
                    if f_active:
                        curr_face_world = transform_camera_to_world(ros_node.face_pose)
                        if prev_face_world is not None:
                            face_speed = np.linalg.norm(curr_face_world - prev_face_world) / dt
                            if face_speed > 0.3: alpha = 0.5 
                            elif face_speed > 0.1: alpha = 0.7 
                        prev_face_world = curr_face_world
                    else:
                        prev_face_world = None
                    
                    if is_dodging: alpha = 0.3 
                    last_time_filter = curr_time

                    for i, j_idx in enumerate(joint_ids):
                        dof_idx = model.joint(joint_names[i]).dofadr[0]
                        target_vel = total_dq[dof_idx] * 0.01
                        
                        filtered_dq[i] = alpha * filtered_dq[i] + (1.0 - alpha) * target_vel
                        
                        if is_dodging: filtered_dq[i] = np.clip(filtered_dq[i], -0.1, 0.1) # 회피 시 속도 제한도 부드럽게 조정
                        else: filtered_dq[i] = np.clip(filtered_dq[i], -0.05, 0.05)
                        
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]

                mujoco.mj_step(model, data)

                if curr_time - last_render_time > (1.0 / 30.0):
                    viewer.user_scn.ngeom = 0
                    
                    # 🛠️ [수정됨] 기존 Box 삭제 후 원통형(Cylinder) 카메라 시각화
                    # 원통의 Z축(기본 축)을 로봇을 향하는 X축으로 눕히기 위한 회전 행렬 적용
                    cam_rot = np.array([[0, 0, 1],
                                        [0, 1, 0],
                                        [-1, 0, 0]])
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_CYLINDER, 
                        [0.025, 0.02, 0.0], # [반지름 2.5cm, 절반 길이 2cm]
                        CAMERA_OFFSET, 
                        cam_rot.flatten(), 
                        [0.2, 0.2, 0.2, 1.0] # 렌즈 같은 진한 회색
                    )
                    viewer.user_scn.ngeom += 1

                    if l_avoid_active: 
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           transform_camera_to_world(ros_node.left_pose), np.eye(3).flatten(), [1, 0, 0, 0.8])
                        viewer.user_scn.ngeom += 1
                    if r_avoid_active: 
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           transform_camera_to_world(ros_node.right_pose), np.eye(3).flatten(), [0, 1, 0, 0.8])
                        viewer.user_scn.ngeom += 1
                    if f_active: 
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           transform_camera_to_world(ros_node.face_pose), np.eye(3).flatten(), [1, 1, 0, 0.8])
                        viewer.user_scn.ngeom += 1
                    viewer.sync()
                    last_render_time = curr_time

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()
