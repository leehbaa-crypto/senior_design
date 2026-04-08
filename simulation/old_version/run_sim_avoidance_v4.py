import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import mujoco
import mujoco.viewer
import numpy as np
import threading
import os
import time

class MuJoCoROS2Node(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.create_subscription(Point, '/target_pose', self.hand_pose_cb, 10)
        self.hand_cam_pose = None
        
    def hand_pose_cb(self, msg):
        self.hand_cam_pose = np.array([msg.x, msg.y, msg.z])

def transform_camera_to_world(cam_pos):
    # 카메라 축 변환 및 오프셋 적용
    world_x, world_y, world_z = cam_pos[2], -cam_pos[0], -cam_pos[1]
    return np.array([world_x, world_y, world_z]) + np.array([0.8, 0.0, 0.5])

def calculate_repulsive_force(ee_pos, hand_pos, threshold=0.4, max_force=2.0):
    direction = ee_pos - hand_pos
    distance = np.linalg.norm(direction)
    if 0.01 < distance < threshold:
        force_magnitude = max_force * ((1.0 / distance) - (1.0 / threshold))
        return (direction / distance) * force_magnitude, distance
    return np.zeros(3), distance

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
        
        # 제어할 관절 이름 (사용 중인 모델에 맞게 설정됨)
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joint_ids = [model.joint(name).qposadr[0] for name in joint_names]
        
        # 끝단(EE) ID 가져오기
        ee_site_name = "gripperframe"
        try:
            ee_id = model.site(ee_site_name).id
            is_site = True
        except KeyError:
            ee_id = model.body("gripper").id
            is_site = False

        # 시뮬레이션 시작 시점의 모델 초기화를 위한 1회 스텝
        mujoco.mj_step(model, data)
        
        # [기능 1] 초기 상태 기억하기 (시각화 정보 없을 때 가만히 있을 위치)
        initial_q = np.array([data.qpos[idx] for idx in joint_ids])
        anchor_pos = data.site_xpos[ee_id].copy() if is_site else data.xpos[ee_id].copy()
        
        # 제어용 변수
        jacp = np.zeros((3, model.nv)) 
        jacr = np.zeros((3, model.nv)) 
        smoothed_q = initial_q.copy() # 현재 로봇의 목표 관절 각도 (적분기 역할)
        
        last_print_time = time.time()
        last_render_time = time.time()
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🟢 Active Tracking & Avoidance Node Running (ikpy Removed!)")
            
            while viewer.is_running():
                # Race Condition 방지를 위한 스냅샷
                current_hand_pose = ros_node.hand_cam_pose 
                dist = 0.0 
                
                # [기능 1] 시각화 정보가 없을 때: 초기 자세 유지
                if current_hand_pose is None:
                    for i, idx in enumerate(joint_ids):
                        data.ctrl[i] = initial_q[i]
                        smoothed_q[i] = initial_q[i] # 누적 각도 초기화
                
                # [기능 2] 시각화 정보가 들어왔을 때: 회피 및 평행(LookAt) 유지
                else:
                    hand_world_pos = transform_camera_to_world(current_hand_pose)
                    ee_pos = data.site_xpos[ee_id] if is_site else data.xpos[ee_id]
                    curr_ori_mat = data.site_xmat[ee_id].reshape(3, 3) if is_site else data.xmat[ee_id].reshape(3, 3)
                    
                    # --- A. 위치 제어 (제자리 유지 인력 + 손 회피 척력) ---
                    # 로봇이 도망가기만 하면 무한히 멀어지므로, 원래 있던 허공(anchor_pos)으로 돌아가려는 힘 추가
                    F_attract = 1.0 * (anchor_pos - ee_pos) 
                    F_rep, dist = calculate_repulsive_force(ee_pos, hand_world_pos, threshold=0.5, max_force=3.0)
                    F_total = F_attract + F_rep

                    # --- B. 방향 제어 (카메라가 손을 정면으로 바라보기) ---
                    # 1. 카메라가 향해야 할 Z축 벡터 계산
                    z_des = hand_world_pos - ee_pos
                    norm_z = np.linalg.norm(z_des)
                    
                    if norm_z > 0.01:
                        z_des = z_des / norm_z
                        
                        # 2. 로봇이 꼬이지 않도록 위쪽(Up) 벡터를 기준으로 X, Y축 직교 생성
                        up = np.array([0.0, 0.0, 1.0])
                        # 만약 손이 로봇 머리 위나 바로 아래에 있어 Z축과 겹치면 Up 벡터 임시 변경
                        if abs(np.dot(z_des, up)) > 0.99: 
                            up = np.array([1.0, 0.0, 0.0]) 
                            
                        x_des = np.cross(up, z_des)
                        x_des = x_des / np.linalg.norm(x_des)
                        y_des = np.cross(z_des, x_des)
                        
                        # 3. 목표 회전 행렬(LookAt) 완성
                        R_des = np.column_stack((x_des, y_des, z_des))
                        
                        # 4. 현재 방향과 목표 방향 간의 오차(Error) 계산
                        err_x = np.cross(curr_ori_mat[:, 0], R_des[:, 0])
                        err_y = np.cross(curr_ori_mat[:, 1], R_des[:, 1])
                        err_z = np.cross(curr_ori_mat[:, 2], R_des[:, 2])
                        e_ori = (err_x + err_y + err_z) / 2.0
                        
                        tau_ori = 1.5 * e_ori # 회전 토크 계수
                    else:
                        tau_ori = np.zeros(3)

                    # --- C. 힘과 토크를 관절 각도로 변환 ---
                    if is_site: mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
                    else: mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
                    
                    # 자코비안을 이용한 역기구학 근사 제어 (ikpy 대체)
                    tau_joint = (jacp.T @ F_total) + (jacr.T @ tau_ori)
                    
                    for i, j_idx in enumerate(joint_ids):
                        # 각 관절에 필요한 미세 움직임량 계산 (0.01은 반응 속도 계수)
                        target_dq = tau_joint[model.joint(joint_names[i]).dofadr[0]] * 0.01
                        smoothed_q[i] += target_dq
                        data.ctrl[i] = smoothed_q[i]

                # 시뮬레이션 진행
                mujoco.mj_step(model, data)

                # 프레임 제한 (30FPS) 및 시각화
                curr_time = time.time()
                if curr_time - last_render_time > (1.0 / 30.0):
                    if current_hand_pose is not None:
                        viewer.user_scn.ngeom = 0
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           hand_world_pos, np.eye(3).flatten(), [1, 0, 0, 0.8])
                        viewer.user_scn.ngeom += 1
                    viewer.sync()
                    last_render_time = curr_time

                # 로그 출력
                if curr_time - last_print_time > 1.0 and current_hand_pose is not None:
                    print(f"📏 손과의 거리: {dist:.2f}m | 👀 카메라 타겟 응시 중")
                    last_print_time = curr_time

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()
