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
CAMERA_OFFSET = np.array([0.25, -0.15, 0.55]) 

AVOIDANCE_THRESHOLD = 0.35  
MAX_REPULSIVE_FORCE = 2.5   
VIRTUAL_PADDING = 0.03      

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
        
        # 🛠️ [떨림 방지 변수] 이전 프레임의 상태를 기억하여 부드럽게 섞기 위함
        smoothed_danger = 0.0
        smoothed_F_rep = np.zeros(3)
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🚀 Active Node v13 Running: Anti-Jitter Smooth Blending")
            
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
                    smoothed_danger = 0.0
                    smoothed_F_rep = np.zeros(3)
                else:
                    ee_pos = data.site_xpos[ee_id] if is_site else data.xpos[ee_id]
                    curr_ori_mat = data.site_xmat[ee_id].reshape(3, 3) if is_site else data.xmat[ee_id].reshape(3, 3)
                    
                    active_hands = []
                    if l_avoid_active: active_hands.append(transform_camera_to_world(ros_node.left_pose))
                    if r_avoid_active: active_hands.append(transform_camera_to_world(ros_node.right_pose))

                    closest_hand_pos = None
                    min_global_dist = 999.0

                    # 🛠️ [떨림 방지 핵심 1] 위험도(Danger Level)를 0.0 ~ 1.0 사이의 연속적인 값으로 계산
                    raw_danger = 0.0
                    raw_F_rep = np.zeros(3)
                    
                    for hand_pos in active_hands:
                        dist_to_ee = np.linalg.norm(ee_pos - hand_pos)
                        if dist_to_ee < min_global_dist:
                            min_global_dist = dist_to_ee
                            closest_hand_pos = hand_pos
                            
                        eff_dist = max(0.01, dist_to_ee - VIRTUAL_PADDING)
                        if eff_dist < AVOIDANCE_THRESHOLD:
                            # 0.35m일 때 danger = 0.0, 완전 닿았을 때 danger = 1.0
                            danger = (AVOIDANCE_THRESHOLD - eff_dist) / AVOIDANCE_THRESHOLD
                            raw_danger = max(raw_danger, danger)
                            
                            force_mag = MAX_REPULSIVE_FORCE * (danger**2)
                            raw_F_rep += ((ee_pos - hand_pos) / dist_to_ee) * force_mag

                    # 🛠️ [떨림 방지 핵심 2] 로우패스 필터(EMA) 적용하여 값이 튀는 것을 방지
                    filter_weight = 0.85 # 이전 값을 85% 유지 (부드러움 결정)
                    smoothed_danger = filter_weight * smoothed_danger + (1.0 - filter_weight) * raw_danger
                    smoothed_F_rep = filter_weight * smoothed_F_rep + (1.0 - filter_weight) * raw_F_rep

                    # 🛠️ [떨림 방지 핵심 3] 스위치가 아닌 비율(Ratio)로 모든 파라미터 조절
                    attract_gain = 0.5 - (0.45 * smoothed_danger) # 위험할수록 0.5 -> 0.05 로 스르륵 감소
                    posture_gain = 0.5 + (1.0 * smoothed_danger)  # 위험할수록 0.5 -> 1.5 로 스르륵 증가
                    dynamic_lambda_sq = 0.01 + (0.09 * smoothed_danger) # 위험할수록 DLS 감쇠를 늘려 특이점 떨림 방지

                    target_look_pos = transform_camera_to_world(ros_node.face_pose) if f_active else closest_hand_pos
                    V_err = np.zeros(6) 
                    
                    F_attract = attract_gain * (anchor_pos - ee_pos) 
                    V_err[0:3] = F_attract + smoothed_F_rep
                    
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
                    
                    # 💡 감쇠(damping) 수치가 가변적으로 적용됨
                    J_pinv = J_ee.T @ np.linalg.inv(J_ee @ J_ee.T + dynamic_lambda_sq * np.eye(6))
                    
                    dq_look = J_pinv @ V_err
                    
                    dq_posture = np.zeros(model.nv)
                    if f_active:
                        bias_shoulder = -1.0 
                        bias_elbow = -1.5    
                        bias_wrist = 1.0     
                        
                        idx_shoulder = model.joint("shoulder_lift").dofadr[0]
                        idx_elbow = model.joint("elbow_flex").dofadr[0]
                        idx_wrist = model.joint("wrist_flex").dofadr[0]
                        
                        dq_posture[idx_shoulder] = (initial_q[1] + bias_shoulder - smoothed_q[1]) * posture_gain
                        dq_posture[idx_elbow] = (initial_q[2] + bias_elbow - smoothed_q[2]) * posture_gain
                        dq_posture[idx_wrist] = (initial_q[3] + bias_wrist - smoothed_q[3]) * posture_gain

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
                    
                    # 회피 상황일 때 필터값 및 속도제한도 부드럽게 전환
                    alpha = alpha - (0.6 * smoothed_danger) # 0.9 -> 최저 0.3
                    alpha = max(0.1, alpha) 
                    
                    dynamic_speed_limit = 0.05 + (0.05 * smoothed_danger) # 최대 속도제한 0.05 -> 0.10 스무스하게
                    
                    last_time_filter = curr_time

                    for i, j_idx in enumerate(joint_ids):
                        dof_idx = model.joint(joint_names[i]).dofadr[0]
                        target_vel = total_dq[dof_idx] * 0.01
                        
                        filtered_dq[i] = alpha * filtered_dq[i] + (1.0 - alpha) * target_vel
                        filtered_dq[i] = np.clip(filtered_dq[i], -dynamic_speed_limit, dynamic_speed_limit) 
                        
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]

                mujoco.mj_step(model, data)

                if curr_time - last_render_time > (1.0 / 30.0):
                    viewer.user_scn.ngeom = 0

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
