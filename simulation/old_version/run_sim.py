import mujoco
import mujoco.viewer
import os
import time

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
scene_path = os.path.join(current_dir, "scene.xml")

print(f"Loading MuJoCo scene from: {scene_path}")

try:
    # 1. MJCF (XML) 파일을 MuJoCo 모델로 로드
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # 2. 시각화를 위한 뷰어 실행
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo Viewer is running! Press Ctrl+C to stop.")
        
        # 간단한 제어 루프
        while viewer.is_running():
            step_start = time.time()

            # 물리 엔진 스텝 진행
            mujoco.mj_step(model, data)

            # 뷰어 업데이트
            viewer.sync()

            # 실시간 동기화 (60FPS 수준)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    print(f"Error loading model: {e}")
    print("\n[TIP] 만약 STL 파일을 못 찾는다면 URDF 내의 <mesh filename='../meshes/...'> 경로를 다시 확인해야 합니다.")
