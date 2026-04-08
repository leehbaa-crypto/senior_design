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
    """
    [핵심 수정] RealSense 좌표계를 MuJoCo 좌표계로 변환
    가정: 카메라는 로봇 정면(X축 방향)에서 로봇을 바라보고 있음
    """
    cam_x, cam_y, cam_z = cam_pos
    
    # 1. 축 변환 (Axis Swapping)
    # 카메라의 Z(깊이) -> 로봇의 X (앞뒤)
    # 카메라의 X(우측) -> 로봇의 Y (좌우, 반전 필요할 수 있음)
    # 카메라의 Y(아래) -> 로봇의 Z (위아래, 반전 필요)
    world_x = cam_z      
    world_y = -cam_x     
    world_z = -cam_y     
    
    # 2. 오프셋 (카메라가 로봇 베이스 기준 어디에 있는지)
    # 예: 로봇 앞으로 0.8m, 높이 0.5m에 카메라가 있다고 가정
    offset = np.array([0.8, 0.0, 0.5]) 
    
    return np.array([world_x, world_y, world_z]) + offset

def calculate_repulsive_force(ee_pos, hand_pos, threshold=0.4, max_force=100.0):
    direction = ee_pos - hand_pos
    distance = np.linalg.norm(direction)
    
    if 0.01 < distance < threshold:
        force_magnitude = max_force * (1.0 / distance - 1.0 / threshold)
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
        
        # [수정] 모델에 존재하는 실제 이름으로 변경
        ee_target_name = "gripperframe"
        is_site = True
        try:
            ee_id = model.site(ee_target_name).id
        except KeyError:
            print(f"⚠️ '{ee_target_name}' site가 없습니다. 'gripper' 바디를 사용합니다.")
            ee_target_name = "gripper" 
            is_site = False
            try:
                ee_id = model.body(ee_target_name).id
            except KeyError:
                print("❌ 'gripper'도 찾을 수 없습니다. scene.xml에서 로봇 끝단의 이름을 확인해주세요!")
                return

        # [추가] 내 손 위치를 시각화하기 위한 사이트 ID (없으면 생성은 안 되므로 로그 출력)
        # 만약 모델에 hand_indicator라는 site를 미리 만들어두면 더 좋습니다.
        
        jacp = np.zeros((3, model.nv))
        last_print_time = time.time()
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("🟢 MuJoCo Viewer & ROS2 Avoidance Node Running!")
            
            while viewer.is_running():
                step_start = time.time()

                if ros_node.hand_cam_pose is not None:
                    # 1. 좌표 변환
                    hand_world_pos = transform_camera_to_world(ros_node.hand_cam_pose)
                    
                    # [추가] 뷰어의 디버깅 기능을 활용해 내 손 위치에 빨간 구체 그리기
                    viewer.user_scn.ngeom = 0 # 이전 프레임 지우기
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                       mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.03, 0.03], 
                                       hand_world_pos, np.eye(3).flatten(), [1, 0, 0, 0.8])
                    viewer.user_scn.ngeom += 1
                    
                    # 2. 현재 EE 위치 가져오기
                    if is_site:
                        ee_pos = data.site_xpos[ee_id]
                    else:
                        ee_pos = data.xpos[ee_id]
                    
                    # 3. 척력 계산
                    F_repulsive, dist = calculate_repulsive_force(ee_pos, hand_world_pos)
                    
                    # 디버깅 출력 (1초 간격)
                    if time.time() - last_print_time > 1.0:
                        print(f"🤖 로봇: {ee_pos.round(2)} | ✋ 내 손: {hand_world_pos.round(2)} | 📏 거리: {dist:.2f}m")
                        last_print_time = time.time()

                    if np.any(F_repulsive):
                        if is_site:
                            mujoco.mj_jacSite(model, data, jacp, None, ee_id)
                        else:
                            mujoco.mj_jacBody(model, data, jacp, None, ee_id)
                            
                        tau = jacp.T @ F_repulsive
                        data.qfrc_applied[:] = tau
                    else:
                        data.qfrc_applied[:] = 0.0 

                mujoco.mj_step(model, data)
                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()
        spin_thread.join()

if __name__ == "__main__":
    main()
