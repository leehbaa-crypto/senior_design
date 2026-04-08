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
# 🛠️ [카메라 좌표 캘리브레이션 설정] 🛠️
# ==========================================
# 1. 카메라가 바라보는 방향 설정 (1 또는 -1)
# 1: 카메라가 로봇의 등 뒤에서 로봇의 정면(빨간축)을 함께 바라보는 경우
# -1: 카메라가 로봇 정면에서 사용자와 로봇을 마주보고 있는 경우
CAMERA_DIRECTION = 1

# 2. 로봇 베이스(0,0,0) 기준 카메라의 실제 위치 [X, Y, Z] (단위: m)
# 빨간색 축 정면에 손이 오도록 X값을 조정하세요.
# 예: 로봇 중심보다 빨간축(X) 방향으로 0.2m 앞, 높이(Z) 0.5m에 카메라가 있다면 -> [0.2, 0.0, 0.5]
CAMERA_OFFSET = np.array([0.2, 0.0, 0.5]) 
# ==========================================


class MuJoCoROS2Node(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.create_subscription(Point, '/target_pose', self.hand_pose_cb, 10)
        self.hand_cam_pose = None
        
    def hand_pose_cb(self, msg):
        self.hand_cam_pose = np.array([msg.x, msg.y, msg.z])

def transform_camera_to_world(cam_pos):
    """
    RealSense의 좌표(X: 우측, Y: 아래, Z: 깊이)를 
    MuJoCo의 좌표(X: 빨강, Y: 초록, Z: 파랑)로 변환합니다.
    """
    # 기본적으로 깊이(cam_pos[2])가 빨간축(X축)이 되도록 매핑
    world_x = cam_pos[2] * CAMERA_DIRECTION
    
    # 거울 모드 등 비전 설정에 따라 좌우(Y축)가 반대로 움직인다면 '-cam_pos[0]'의 부호를 바꿔보세요.
    world_y = -cam_pos[0] * CAMERA_DIRECTION 
    
    # 상하 반전 처리
    world_z = -cam_pos[1] 
    
    return np.array([world_x, world_y, world_z]) + CAMERA_OFFSET

def calculate_repulsive_force(ee_pos, hand_pos, threshold=0.5, max_force=3.0):
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
        
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joint_ids = [model.joint(name).qposadr[0] for name in joint_names]
        
        ee_site_name = "gripperframe"
        try:
            ee_id = model.site(ee_site_name).id
            is_site = True
        except KeyError:
            ee_id = model.body("gripper").id
            is_site = False

        mujoco.mj_step(model, data)
        
        # [기능 1] 초기 상태 기억하기
        initial_q = np.array([data.qpos[idx] for idx in joint_ids])
        anchor_pos = data.site_xpos[ee_id].copy() if is_site else data.xpos[ee_id].copy()
        
        jacp = np.zeros((3, model.nv)) 
        jacr = np.zeros((3, model.nv)) 
        smoothed_q = initial_q.copy() 
        
        last_print_time = time.time()
        last_render_time = time.time()
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🟢 Active Tracking & Avoidance Node Running (Calibrated)")
            
            while viewer.is_running():
                current_hand_pose = ros_node.hand_cam_pose 
                dist = 0.0 
                
                # [대기 모드]
                if current_hand_pose is None:
                    for i, idx in enumerate(joint_ids):
                        data.ctrl[i] = initial_q[i]
                        smoothed_q[i] = initial_q[i] 
                
                # [회피 및 응시 모드]
                else:
                    hand_world_pos = transform_camera_to_world(current_hand_pose)
                    ee_pos = data.site_xpos[ee_id] if is_site else data.xpos[ee_id]
                    curr_ori_mat = data.site_xmat[ee_id].reshape(3, 3) if is_site else data.xmat[ee_id].reshape(3, 3)
                    
                    # --- A. 위치 제어 (원위치 복귀 + 회피력) ---
                    F_attract = 1.0 * (anchor_pos - ee_pos) 
                    F_rep, dist = calculate_repulsive_force(ee_pos, hand_world_pos, threshold=0.5, max_force=3.0)
                    F_total = F_attract + F_rep

                    # --- B. 방향 제어 (Look-At) ---
                    z_des = hand_world_pos - ee_pos
                    norm_z = np.linalg.norm(z_des)
                    
                    if norm_z > 0.01:
                        z_des = z_des / norm_z
                        up = np.array([0.0, 0.0, 1.0])
                        if abs(np.dot(z_des, up)) > 0.99: 
                            up = np.array([1.0, 0.0, 0.0]) 
                            
                        x_des = np.cross(up, z_des)
                        x_des = x_des / np.linalg.norm(x_des)
                        y_des = np.cross(z_des, x_des)
                        
                        R_des = np.column_stack((x_des, y_des, z_des))
                        
                        err_x = np.cross(curr_ori_mat[:, 0], R_des[:, 0])
                        err_y = np.cross(curr_ori_mat[:, 1], R_des[:, 1])
                        err_z = np.cross(curr_ori_mat[:, 2], R_des[:, 2])
                        e_ori = (err_x + err_y + err_z) / 2.0
                        
                        tau_ori = 1.5 * e_ori 
                    else:
                        tau_ori = np.zeros(3)

                    # --- C. 역기구학 근사 제어 ---
                    if is_site: mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
                    else: mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
                    
                    tau_joint = (jacp.T @ F_total) + (jacr.T @ tau_ori)
                    
                    for i, j_idx in enumerate(joint_ids):
                        target_dq = tau_joint[model.joint(joint_names[i]).dofadr[0]] * 0.01
                        smoothed_q[i] += target_dq
                        data.ctrl[i] = smoothed_q[i]

                mujoco.mj_step(model, data)

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
