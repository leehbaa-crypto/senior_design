import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import mujoco
import mujoco.viewer
import numpy as np
import threading
import os
import time

CAMERA_DIRECTION = 1
CAMERA_OFFSET = np.array([0.2, 0.0, 0.5]) 

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

def calculate_repulsive_force(ee_pos, hand_pos, threshold=0.5, max_force=2.0):
    direction = ee_pos - hand_pos
    distance = np.linalg.norm(direction)
    if 0.01 < distance < threshold:
        force_magnitude = max_force * ((threshold - distance) / threshold)**2
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
            ee_id = model.site(ee_site_name).id; is_site = True
        except KeyError:
            ee_id = model.body("gripper").id; is_site = False

        mujoco.mj_step(model, data)
        initial_q = np.array([data.qpos[idx] for idx in joint_ids])
        anchor_pos = data.site_xpos[ee_id].copy() if is_site else data.xpos[ee_id].copy()
        
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv)) 
        smoothed_q = initial_q.copy() 
        filtered_dq = np.zeros(len(joint_ids))
        
        last_render_time = time.time()
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🟢 Active Node Running: Red Axis (X) Looks at Face")
            
            while viewer.is_running():
                curr_time = time.time()
                
                l_active = (curr_time - ros_node.left_time) < 0.5 and ros_node.left_pose is not None
                r_active = (curr_time - ros_node.right_time) < 0.5 and ros_node.right_pose is not None
                f_active = (curr_time - ros_node.face_time) < 0.5 and ros_node.face_pose is not None
                
                # [대기 모드] 부드러운 복귀
                if not l_active and not r_active and not f_active:
                    for i, idx in enumerate(joint_ids):
                        target_dq = (initial_q[i] - smoothed_q[i]) * 0.03
                        filtered_dq[i] = 0.9 * filtered_dq[i] + 0.1 * target_dq
                        filtered_dq[i] = np.clip(filtered_dq[i], -0.05, 0.05)
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]
                else:
                    ee_pos = data.site_xpos[ee_id] if is_site else data.xpos[ee_id]
                    curr_ori_mat = data.site_xmat[ee_id].reshape(3, 3) if is_site else data.xmat[ee_id].reshape(3, 3)
                    
                    F_total = 1.0 * (anchor_pos - ee_pos) 
                    min_dist = 999.0
                    closest_hand_pos = None

                    # 회피력 계산
                    if l_active:
                        l_world = transform_camera_to_world(ros_node.left_pose)
                        F, d = calculate_repulsive_force(ee_pos, l_world)
                        F_total += F
                        if d < min_dist: 
                            min_dist = d
                            closest_hand_pos = l_world

                    if r_active:
                        r_world = transform_camera_to_world(ros_node.right_pose)
                        F, d = calculate_repulsive_force(ee_pos, r_world)
                        F_total += F
                        if d < min_dist: 
                            min_dist = d
                            closest_hand_pos = r_world

                    target_look_pos = None
                    if f_active:
                        target_look_pos = transform_camera_to_world(ros_node.face_pose)
                    elif closest_hand_pos is not None:
                        target_look_pos = closest_hand_pos

                    # [핵심 수정] X축(빨간색 축)이 목표물을 바라보도록 수학 행렬 교체
                    if target_look_pos is not None:
                        x_des = target_look_pos - ee_pos
                        norm_x = np.linalg.norm(x_des)
                        if norm_x > 0.01:
                            x_des = x_des / norm_x
                            
                            # 로봇이 꼬이지 않도록 업(Up) 벡터 설정
                            up = np.array([0.0, 0.0, 1.0]) 
                            # 만약 로봇이 정수리 위나 발밑을 쳐다봐서 짐벌락이 오면 업 벡터 임시 변경
                            if abs(np.dot(x_des, up)) > 0.99: 
                                up = np.array([0.0, 1.0, 0.0]) 
                                
                            # 직교하는 Y, Z축 생성
                            y_des = np.cross(up, x_des)
                            y_des = y_des / np.linalg.norm(y_des)
                            z_des = np.cross(x_des, y_des)
                            
                            # 빨간축(X)이 타겟을 향하는 회전 행렬 완성
                            R_des = np.column_stack((x_des, y_des, z_des))
                            
                            err_x = np.cross(curr_ori_mat[:, 0], R_des[:, 0])
                            err_y = np.cross(curr_ori_mat[:, 1], R_des[:, 1])
                            err_z = np.cross(curr_ori_mat[:, 2], R_des[:, 2])
                            
                            tau_ori = 1.0 * ((err_x + err_y + err_z) / 2.0)
                        else:
                            tau_ori = np.zeros(3)
                    else:
                        tau_ori = np.zeros(3)

                    # 관절 제어 입력 계산
                    if is_site: mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
                    else: mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
                    
                    tau_joint = (jacp.T @ F_total) + (jacr.T @ tau_ori)
                    
                    for i, j_idx in enumerate(joint_ids):
                        target_dq = tau_joint[model.joint(joint_names[i]).dofadr[0]] * 0.005 
                        
                        filtered_dq[i] = 0.9 * filtered_dq[i] + 0.1 * target_dq
                        filtered_dq[i] = np.clip(filtered_dq[i], -0.05, 0.05)
                        
                        smoothed_q[i] += filtered_dq[i]
                        data.ctrl[i] = smoothed_q[i]

                mujoco.mj_step(model, data)

                # 시각화 (30FPS)
                if curr_time - last_render_time > (1.0 / 30.0):
                    viewer.user_scn.ngeom = 0
                    if l_active: 
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                           mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                           transform_camera_to_world(ros_node.left_pose), np.eye(3).flatten(), [1, 0, 0, 0.8])
                        viewer.user_scn.ngeom += 1
                    if r_active: 
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
