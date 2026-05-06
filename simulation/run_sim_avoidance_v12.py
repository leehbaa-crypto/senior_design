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
# 🛠️ [설정 값: 카메라 왜곡 완벽 교정] 🛠️
# ==========================================
CAMERA_DIRECTION = 1
CAMERA_OFFSET = np.array([0.25, -0.15, 0.50]) 

# 1. 좌우/상하 반전 교정 (오른손이 왼쪽으로 갔던 문제 해결)
CAMERA_Y_SIGN = -1  # 좌우 반전 해결
CAMERA_Z_SIGN = -1  # 상하 반전 제어

# 2. 높이 왜곡 해결 (카메라가 아래를 쳐다보는 각도 보정)

CAMERA_PITCH_DEG = 0.0  

AVOIDANCE_THRESHOLD = 0.4   
MAX_REPULSIVE_FORCE = 4.0   
VIRTUAL_PADDING = 0.05      

class MuJoCoROS2Node(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.create_subscription(Point, '/left_hand', self.left_cb, 10)
        self.create_subscription(Point, '/right_hand', self.right_cb, 10)
        self.create_subscription(Point, '/face_pose', self.face_cb, 10)
        
        self.left_pose = None; self.left_time = 0.0
        self.right_pose = None; self.right_time = 0.0
        self.face_pose = None; self.face_time = 0.0
        
    def is_valid(self, msg):
        return not (abs(msg.x) < 0.001 and abs(msg.y) < 0.001 and abs(msg.z) < 0.001)

    def left_cb(self, msg):
        if self.is_valid(msg): 
            self.left_pose = np.array([msg.x, msg.y, msg.z]); self.left_time = time.time()
    def right_cb(self, msg):
        if self.is_valid(msg):
            self.right_pose = np.array([msg.x, msg.y, msg.z]); self.right_time = time.time()
    def face_cb(self, msg):
        if self.is_valid(msg):
            self.face_pose = np.array([msg.x, msg.y, msg.z]); self.face_time = time.time()

def transform_camera_to_world(cam_pos):
    raw_x = cam_pos[2] * CAMERA_DIRECTION
    raw_y = cam_pos[0] * CAMERA_Y_SIGN 
    raw_z = cam_pos[1] * CAMERA_Z_SIGN  
    
    pitch = np.radians(CAMERA_PITCH_DEG)
    R_y = np.array([
        [np.cos(pitch),  0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    rotated_pos = R_y @ np.array([raw_x, raw_y, raw_z])
    return rotated_pos + CAMERA_OFFSET

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
        for _ in range(50): mujoco.mj_step(model, data) 
        
        initial_q = np.array([data.qpos[idx] for idx in joint_ids])
        anchor_pos = data.site_xpos[ee_id].copy() if is_site else data.xpos[ee_id].copy()
        
        smoothed_q = initial_q.copy() 
        filtered_dq = np.zeros(len(joint_ids))
        
        last_render_time = time.time()
        last_time_filter = time.time()
        
        # 💡 [버그 완벽 해결] 파이썬 에러의 원인이었던 변수 초기화 부활!
        last_known_face_pos = None 
        prev_face_world = None 
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🚀 Active Node Running: Bug Fixed & System Stable!")
            
            while viewer.is_running():
                curr_time = time.time()
                dt = curr_time - last_time_filter
                if dt <= 0: dt = 0.001
                
                l_avoid_active = (curr_time - ros_node.left_time) < 1.0 and ros_node.left_pose is not None
                r_avoid_active = (curr_time - ros_node.right_time) < 1.0 and ros_node.right_pose is not None
                f_active = (curr_time - ros_node.face_time) < 0.5 and ros_node.face_pose is not None
                
                if not l_avoid_active and not r_avoid_active and not f_active:
                    for i, idx in enumerate(joint_ids):
                        filtered_dq[i] = 0.8 * filtered_dq[i] 
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]
                        
                    # 💡 놓쳤을 때도 변수 초기화
                    prev_face_world = None
                else:
                    ee_pos = data.site_xpos[ee_id] if is_site else data.xpos[ee_id]
                    curr_ori_mat = data.site_xmat[ee_id].reshape(3, 3) if is_site else data.xmat[ee_id].reshape(3, 3)
                    
                    render_l_hand = (curr_time - ros_node.left_time) < 0.5 and ros_node.left_pose is not None
                    render_r_hand = (curr_time - ros_node.right_time) < 0.5 and ros_node.right_pose is not None
                    
                    active_hands = []
                    if l_avoid_active: active_hands.append(transform_camera_to_world(ros_node.left_pose))
                    if r_avoid_active: active_hands.append(transform_camera_to_world(ros_node.right_pose))

                    if f_active:
                        raw_face_pos = transform_camera_to_world(ros_node.face_pose)
                        if last_known_face_pos is None:
                            last_known_face_pos = raw_face_pos
                        else:
                            last_known_face_pos = 0.85 * last_known_face_pos + 0.15 * raw_face_pos

                    dq_avoid = np.zeros(model.nv)
                    F_repulsive_ee = np.zeros(3)
                    closest_hand_pos = None
                    min_global_dist = 999.0
                    is_dodging = False

                    for hand_pos in active_hands:
                        dist_to_ee = np.linalg.norm(ee_pos - hand_pos)
                        if dist_to_ee < min_global_dist:
                            min_global_dist = dist_to_ee
                            closest_hand_pos = hand_pos
                            
                        eff_dist_ee = max(0.01, dist_to_ee - VIRTUAL_PADDING)
                        if eff_dist_ee < AVOIDANCE_THRESHOLD:
                            is_dodging = True
                            danger = (AVOIDANCE_THRESHOLD - eff_dist_ee) / AVOIDANCE_THRESHOLD
                            force_mag = MAX_REPULSIVE_FORCE * (danger**2)
                            
                            direction = ee_pos - hand_pos
                            direction[2] = max(0.0, direction[2]) 
                            if np.linalg.norm(direction) > 0.001:
                                direction = direction / np.linalg.norm(direction)
                            F_repulsive_ee += direction * force_mag

                        for body_id in range(1, model.nbody):
                            body_pos = data.xpos[body_id]
                            direction = body_pos - hand_pos
                            direction[2] = max(0.0, direction[2]) 
                            dist = np.linalg.norm(direction)
                            if dist == 0: continue
                                
                            eff_dist = max(0.01, dist - VIRTUAL_PADDING)
                            if eff_dist < AVOIDANCE_THRESHOLD:
                                force_mag = MAX_REPULSIVE_FORCE * ((AVOIDANCE_THRESHOLD - eff_dist) / AVOIDANCE_THRESHOLD)**3
                                F_rep = (direction / dist) * force_mag
                                jacp_b = np.zeros((3, model.nv))
                                mujoco.mj_jacBody(model, data, jacp_b, None, body_id)
                                dq_avoid += (jacp_b.T @ F_rep) * 0.015 

                    target_look_pos = last_known_face_pos if last_known_face_pos is not None else closest_hand_pos
                    V_err = np.zeros(6) 
                    
                    active_target_pos = anchor_pos.copy()
                    if f_active and target_look_pos is not None:
                        active_target_pos[0] -= 0.15 
                        active_target_pos[2] += 0.10 

                    attract_gain = 0.05 if is_dodging else 0.2 
                    pos_error = active_target_pos - ee_pos
                    if np.linalg.norm(pos_error[0:2]) < 0.02: pos_error[0:2] = 0.0 
                    
                    F_attract = attract_gain * pos_error 
                    
                    V_err[0] = F_attract[0] + F_repulsive_ee[0]
                    V_err[1] = F_attract[1] + F_repulsive_ee[1]
                    V_err[2] = F_attract[2] + F_repulsive_ee[2]
                    
                    if ee_pos[2] <= anchor_pos[2]: 
                        V_err[2] = max(0.0, V_err[2])
                    if ee_pos[0] >= anchor_pos[0]:
                        V_err[0] = min(0.0, V_err[0])
                    
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
                            
                            e_ori = (err_x + err_y + err_z) / 2.0
                            if np.linalg.norm(e_ori) < 0.03: e_ori = np.zeros(3) 
                            V_err[3:6] = 1.5 * e_ori

                    jacp_ee = np.zeros((3, model.nv))
                    jacr_ee = np.zeros((3, model.nv))
                    if is_site: mujoco.mj_jacSite(model, data, jacp_ee, jacr_ee, ee_id)
                    else: mujoco.mj_jacBody(model, data, jacp_ee, jacr_ee, ee_id)
                    J_ee = np.vstack((jacp_ee, jacr_ee))
                    
                    lambda_sq = 0.06 
                    J_pinv = J_ee.T @ np.linalg.inv(J_ee @ J_ee.T + lambda_sq * np.eye(6))
                    
                    dq_look = J_pinv @ V_err
                    total_dq = dq_look + dq_avoid

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
                        
                        if is_dodging:
                            filtered_dq[i] = np.clip(filtered_dq[i], -0.15, 0.15) 
                        else:
                            filtered_dq[i] = np.clip(filtered_dq[i], -0.05, 0.05) 
                        
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]

                mujoco.mj_step(model, data)

                if curr_time - last_render_time > (1.0 / 30.0):
                    viewer.user_scn.ngeom = 0
                    
                    if render_l_hand: 
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           transform_camera_to_world(ros_node.left_pose), np.eye(3).flatten(), [1, 0, 0, 0.8])
                        viewer.user_scn.ngeom += 1
                    if render_r_hand: 
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           transform_camera_to_world(ros_node.right_pose), np.eye(3).flatten(), [0, 1, 0, 0.8])
                        viewer.user_scn.ngeom += 1
                    
                    vis_face_pos = transform_camera_to_world(ros_node.face_pose) if f_active else last_known_face_pos
                    if vis_face_pos is not None:
                        alpha_color = 0.8 if f_active else 0.3
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           vis_face_pos, np.eye(3).flatten(), [1, 1, 0, alpha_color])
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
