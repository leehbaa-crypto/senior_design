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
        # senior_realsense_mediapipe.py가 /target_pose로 손 위치를 보냅니다.
        self.create_subscription(Point, '/target_pose', self.hand_pose_cb, 10)
        self.hand_cam_pose = None
        
    def hand_pose_cb(self, msg):
        self.hand_cam_pose = np.array([msg.x, msg.y, msg.z])

def transform_camera_to_world(cam_pos):
    """
    RealSense 좌표계 (X-right, Y-down, Z-forward) -> MuJoCo (X-forward, Y-left, Z-up)
    """
    cam_x, cam_y, cam_z = cam_pos
    
    # 1. 축 변환
    world_x = cam_z      
    world_y = -cam_x     
    world_z = -cam_y     
    
    # 2. 카메라 베이스 위치 오프셋 (run_sim_avoidance.py와 동일 설정)
    offset = np.array([0.8, 0.0, 0.5]) 
    
    return np.array([world_x, world_y, world_z]) + offset

def calculate_repulsive_force(ee_pos, hand_pos, threshold=0.3, max_force=0.5):
    """
    APF: 장애물(손)로부터 멀어지려는 척력 계산
    """
    direction = ee_pos - hand_pos
    distance = np.linalg.norm(direction)
    
    if 0.01 < distance < threshold:
        # 거리에 반비례하는 힘 (단위: m/step 혹은 가중치)
        force_magnitude = max_force * (1.0 / distance - 1.0 / threshold)
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
    
    # 1. ikpy 체인 로드
    # URDF에서 'base_link'부터 'gripper_frame_link'까지의 체인을 만듭니다.
    my_chain = ikpy.chain.Chain.from_urdf_file(urdf_path, active_links_mask=[False, True, True, True, True, True, False])

    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        # 제어할 조인트 인덱스 (URDF 순서와 MuJoCo 순서 매칭 확인 필요)
        # MuJoCo joint names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joint_ids = [model.joint(name).qposadr[0] for name in joint_names]
        
        # End-Effector 사이트 ID
        ee_site_name = "gripperframe"
        try:
            ee_id = model.site(ee_site_name).id
        except KeyError:
            ee_site_name = "gripper"
            ee_id = model.body(ee_site_name).id

        # 목표 위치 (Goal) - 초기 설정
        goal_pos = np.array([0.2, 0.0, 0.3])
        
        last_print_time = time.time()
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"🟢 IK + APF Avoidance Node Running!")
            print(f"Goal Position: {goal_pos}")
            
            while viewer.is_running():
                step_start = time.time()

                # 1. 현재 조인트 상태 가져오기
                q_current = [0.0] + [data.qpos[idx] for idx in joint_ids] + [0.0] # ikpy 체인 길이에 맞춤
                
                # 2. 목표 IK 계산 (Goal Reaching)
                # ikpy는 4x4 matrix나 [x, y, z]를 받습니다.
                ik_joints = my_chain.inverse_kinematics(goal_pos)
                
                # 3. 손(장애물) 위치 처리 및 회피 (Avoidance)
                if ros_node.hand_cam_pose is not None:
                    hand_world_pos = transform_camera_to_world(ros_node.hand_cam_pose)
                    
                    # 시각화 (빨간 구체)
                    viewer.user_scn.ngeom = 0
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                       mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                       hand_world_pos, np.eye(3).flatten(), [1, 0, 0, 0.8])
                    viewer.user_scn.ngeom += 1
                    
                    # EE 위치 가져오기
                    ee_pos = data.site_xpos[ee_id] if "site" in str(type(model.site)) else data.xpos[ee_id]
                    
                    # 척력 계산
                    F_rep, dist = calculate_repulsive_force(ee_pos, hand_world_pos)
                    
                    if np.any(F_rep):
                        # 자코비안을 이용해 척력을 조인트 공간의 변화량으로 변환
                        jacp = np.zeros((3, model.nv))
                        mujoco.mj_jacSite(model, data, jacp, None, ee_id)
                        
                        # delta_q = J^T * F_rep (간단한 형태)
                        # 여기서는 IK 결과값에 오프셋을 주는 방식으로 구현
                        dq_avoid = jacp.T @ F_rep
                        # 제어 대상 조인트들만 추출 (nv 차원 -> 5개 조인트)
                        # data.ctrl (actuator)에 적용하기 위해 매핑
                        for i, j_idx in enumerate(joint_ids):
                            # Position Control이므로 target angle을 수정
                            # ik_joints[i+1]은 ikpy에서 계산된 해당 조인트의 목표 각도
                            data.ctrl[i] = ik_joints[i+1] + dq_avoid[model.joint(joint_names[i]).dofadr[0]] * 0.1
                    else:
                        for i in range(len(joint_ids)):
                            data.ctrl[i] = ik_joints[i+1]
                            
                    if time.time() - last_print_time > 1.0:
                        print(f"📏 Distance to Hand: {dist:.2f}m | EE Pos: {ee_pos.round(2)}")
                        last_print_time = time.time()
                else:
                    # 손이 감지 안될 때는 그냥 IK 목표로 이동
                    for i in range(len(joint_ids)):
                        data.ctrl[i] = ik_joints[i+1]

                mujoco.mj_step(model, data)
                viewer.sync()

                # FPS 유지
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()
