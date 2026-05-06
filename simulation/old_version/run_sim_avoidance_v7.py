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
# 사용자가 로봇을 마주볼 때, 그리퍼 우측(-Y) & 살짝 뒤(-X)에 안 겹치게 배치
# 예: 그리퍼 기본 위치가 X=0.4일 때, X=0.25(뒤로), Y=-0.15(사용자 기준 우측), Z=0.55(살짝 위)
CAMERA_OFFSET = np.array([0.25, -0.15, 0.55]) 

AVOIDANCE_THRESHOLD = 0.5   # 회피 시작 거리
MAX_REPULSIVE_FORCE = 5.0   # 밀어내는 힘 (엔드이펙터 직접 타격)
VIRTUAL_PADDING = 0.1       # 10cm 가상 보호막 (실제 닿기 전에 피함)

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
            print("🚀 Active Node v10 Running: Visible Camera & Task-Space Retreat Avoidance!")
            
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

                    # ==========================================
                    # 🛠️ [핵심 1] 엔드이펙터(그리퍼) 전용 강력한 Task-Space 회피력 계산
                    # ==========================================
                    F_repulsive_ee = np.zeros(3)
                    for hand_pos in active_hands:
                        # 시선 추적을 위한 가장 가까운 손 찾기
                        dist_to_ee = np.linalg.norm(ee_pos - hand_pos)
                        if dist_to_ee < min_global_dist:
                            min_global_dist = dist_to_ee
                            closest_hand_pos = hand_pos
                            
                        # 가상 보호막 10cm 적용
                        eff_dist = max(0.01, dist_to_ee - VIRTUAL_PADDING)
                        if eff_dist < AVOIDANCE_THRESHOLD:
                            is_dodging = True
                            # 거리가 가까울수록 2제곱으로 튕겨내는 힘
                            force_mag = MAX_REPULSIVE_FORCE * ((AVOIDANCE_THRESHOLD - eff_dist) / AVOIDANCE_THRESHOLD)**2
                            F_repulsive_ee += ((ee_pos - hand_pos) / dist_to_ee) * force_mag

                    # ==========================================
                    # 🛠️ [핵심 2] 역기구학(IK)에 회피 명령을 직접 삽입
                    # ==========================================
                    target_look_pos = transform_camera_to_world(ros_node.face_pose) if f_active else closest_hand_pos
                    V_err = np.zeros(6) 
                    
                    # 원래 자리로 돌아가려는 힘 (약하게 유지)
                    F_attract = 0.5 * (anchor_pos - ee_pos) 
                    
                    # 💡 위치 제어 목표 = (원위치 복귀 힘) + (손을 피해 도망가는 힘)
                    # 이렇게 하면 쳐다보는 각도는 유지하면서 팔 전체가 뒤로 쭉 빠집니다!
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
                    
                    # IK 계산 완료 (시선 유지 + 거리 벌리기 동시 달성)
                    dq_look = J_pinv @ V_err
                    
                    # 자세 유도 (Elbow-Up)
                    dq_posture = np.zeros(model.nv)
                    if f_active and not is_dodging: # 도망칠 때는 자세 유도보다 도망이 우선!
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

                    # 필터 및 속도 제한 로직
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
                    
                    if is_dodging: alpha = 0.3 # 도망갈 땐 즉각 반응
                    last_time_filter = curr_time

                    for i, j_idx in enumerate(joint_ids):
                        dof_idx = model.joint(joint_names[i]).dofadr[0]
                        target_vel = total_dq[dof_idx] * 0.01
                        
                        filtered_dq[i] = alpha * filtered_dq[i] + (1.0 - alpha) * target_vel
                        
                        if is_dodging: filtered_dq[i] = np.clip(filtered_dq[i], -0.2, 0.2) # 스피드 리미트 대폭 해제
                        else: filtered_dq[i] = np.clip(filtered_dq[i], -0.05, 0.05)
                        
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]

                mujoco.mj_step(model, data)

                # ==========================================
                # 🛠️ [핵심 3] 시뮬레이션 내 카메라 및 타겟 시각화
                # ==========================================
                if curr_time - last_render_time > (1.0 / 30.0):
                    viewer.user_scn.ngeom = 0
                    
                    # 1. 인텔 리얼센스 카메라 모델링 (회색 직육면체)
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_BOX, 
                        [0.015, 0.045, 0.015], # 두께, 가로, 세로 비율
                        CAMERA_OFFSET, 
                        np.eye(3).flatten(), 
                        [0.3, 0.3, 0.3, 1.0] # 진회색
                    )
                    viewer.user_scn.ngeom += 1

                    # 2. 손과 얼굴 시각화
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
