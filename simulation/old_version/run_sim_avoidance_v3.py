import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import mujoco
import mujoco.viewer
import numpy as np
import threading
import os
import time
import ikpy.chain

class MuJoCoROS2Node(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.create_subscription(Point, '/target_pose', self.hand_pose_cb, 10)
        self.hand_cam_pose = None
        
    def hand_pose_cb(self, msg):
        self.hand_cam_pose = np.array([msg.x, msg.y, msg.z])

def transform_camera_to_world(cam_pos):
    world_x, world_y, world_z = cam_pos[2], -cam_pos[0], -cam_pos[1]
    return np.array([world_x, world_y, world_z]) + np.array([0.8, 0.0, 0.5])

def calculate_repulsive_force(ee_pos, hand_pos, threshold=0.4, max_force=1.5):
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
    urdf_path = os.path.join(current_dir, "../urdf/so101.urdf")
    scene_path = os.path.join(current_dir, "scene.xml")
    
    my_chain = ikpy.chain.Chain.from_urdf_file(urdf_path, active_links_mask=[False, True, True, True, True, True, False])

    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joint_ids = [model.joint(name).qposadr[0] for name in joint_names]
        
        # Site 또는 Body ID 가져오기
        ee_site_name = "gripperframe"
        try:
            ee_id = model.site(ee_site_name).id
            is_site = True
        except KeyError:
            ee_id = model.body("gripper").id
            is_site = False

        # [최적화 1] IK 연산은 목표가 바뀔 때만 (현재는 정적이므로 시작 전 1회만 계산)
        goal_pos = np.array([0.2, 0.0, 0.3])
        print("⏳ 목표 관절 각도(IK) 계산 중...")
        target_q_full = my_chain.inverse_kinematics(goal_pos)
        target_q = [target_q_full[i+1] for i in range(len(joint_names))]
        print("✅ IK 계산 완료!")

        # 제어용 변수
        jacp = np.zeros((3, model.nv))
        smoothed_dq = np.zeros(len(joint_ids)) # [최적화 2] 움직임을 부드럽게 만드는 로우패스 필터용
        filter_alpha = 0.1 # 필터 강도 (작을수록 부드럽지만 반응이 느려짐)
        
        last_print_time = time.time()
        last_render_time = time.time()
        render_fps = 30.0 # [최적화 3] 렌더링은 30프레임으로 제한
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🟢 Lightweight Avoidance Node Running!")
            
            while viewer.is_running():
                # 1. 척력 계산 및 필터링
                if ros_node.hand_cam_pose is not None:
                    hand_world_pos = transform_camera_to_world(ros_node.hand_cam_pose)
                    
                    # 내 손 위치 그리기
                    viewer.user_scn.ngeom = 0
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                       mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                       hand_world_pos, np.eye(3).flatten(), [1, 0, 0, 0.8])
                    viewer.user_scn.ngeom += 1
                    
                    ee_pos = data.site_xpos[ee_id] if is_site else data.xpos[ee_id]
                    F_rep, dist = calculate_repulsive_force(ee_pos, hand_world_pos)
                    
                    if np.any(F_rep):
                        if is_site: mujoco.mj_jacSite(model, data, jacp, None, ee_id)
                        else: mujoco.mj_jacBody(model, data, jacp, None, ee_id)
                        
                        raw_dq = jacp.T @ F_rep
                        
                        # [제어 최적화] Low-Pass Filter를 통한 부드러운 회피
                        for i, j_idx in enumerate(joint_ids):
                            target_dq = raw_dq[model.joint(joint_names[i]).dofadr[0]] * 0.05 # 회피 민감도
                            smoothed_dq[i] = filter_alpha * target_dq + (1 - filter_alpha) * smoothed_dq[i]
                            
                            data.ctrl[i] = target_q[i] + smoothed_dq[i]
                    else:
                        # 회피 반경 밖이면 부드럽게 원래 위치로 복귀
                        for i in range(len(joint_ids)):
                            smoothed_dq[i] = (1 - filter_alpha) * smoothed_dq[i]
                            data.ctrl[i] = target_q[i] + smoothed_dq[i]
                else:
                    for i in range(len(joint_ids)):
                        data.ctrl[i] = target_q[i]

                # 2. 물리 엔진 스텝 (빠르게 회전)
                mujoco.mj_step(model, data)

                # 3. 뷰어 동기화 (초당 30번만 렌더링하여 CPU 확보)
                curr_time = time.time()
                if curr_time - last_render_time > (1.0 / render_fps):
                    viewer.sync()
                    last_render_time = curr_time

                # 디버깅 출력은 1초에 한 번만
                if curr_time - last_print_time > 1.0 and ros_node.hand_cam_pose is not None:
                    print(f"📏 거리: {dist:.2f}m | 🤖 EE: {ee_pos.round(2)} | ✋ 손: {hand_world_pos.round(2)}")
                    last_print_time = curr_time

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()
