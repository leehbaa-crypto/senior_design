import sys
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import threading
import time

class LightweightMultiTracker(Node):
    def __init__(self):
        super().__init__('lightweight_multi_tracker')
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)

        # MediaPipe 솔루션 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 1. 손 추적기 (v3 로직)
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            model_complexity=0, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.5
        )

        # 2. 얼굴 추적기 (경량화 로직)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False, # [최적화 1] 홍채 및 세부 인식 비활성화로 연산량 대폭 감소
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except Exception as e:
            self.get_logger().error(f"RealSense Error: {e}")
            sys.exit(1)

        self.size_to_depth_c = {"Left": 40.0, "Right": 40.0} 
        
        self.is_running = True
        self.vision_thread = threading.Thread(target=self.camera_loop)
        self.vision_thread.start()
        self.get_logger().info("🚀 Ultra-Light Multi Tracker Started (Hand V3 + Face Contours)")

    def camera_loop(self):
        while self.is_running and rclpy.ok():
            start_time = time.time()

            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame: 
                continue

            image = np.asanyarray(color_frame.get_data())
            display = cv2.flip(image, 1) # 직관성을 위한 거울 모드
            
            # [최적화 2] RGB 변환을 한 번만 수행하여 두 모델(손, 얼굴)이 공유
            rgb_display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            
            hand_results = self.hands.process(rgb_display)
            face_results = self.face_mesh.process(rgb_display)
            
            # --- 1. 얼굴 시각화 (단순 인식 & 렌더링 최적화) ---
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # [최적화 3] 복잡한 그물망 대신 핵심 윤곽선(Contours)만 그려서 UI 렌더링 부하 최소화
                    self.mp_drawing.draw_landmarks(
                        image=display,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

            # --- 2. 양손 추적 및 하이브리드 거리 측정 (v3 로직) ---
            if hand_results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    label = hand_results.multi_handedness[idx].classification[0].label 
                    
                    cx = int(hand_landmarks.landmark[9].x * 640)
                    cy = int(hand_landmarks.landmark[9].y * 480)
                    wx = int(hand_landmarks.landmark[0].x * 640)
                    wy = int(hand_landmarks.landmark[0].y * 480)

                    if 20 < cx < 620 and 20 < cy < 460: 
                        hand_len_px = max(5.0, np.sqrt((cx - wx)**2 + (cy - wy)**2))
                        
                        depth_x = min(639, max(0, 639 - cx)) # 거울 모드 좌표 반전 보정
                        raw_dist = depth_frame.get_distance(depth_x, cy)
                        
                        final_dist = 0.0
                        dist_source = "Vision"

                        if 0.35 < raw_dist < 1.2 and hand_len_px > 40:
                            final_dist = raw_dist
                            dist_source = "LiDAR"
                            current_c = raw_dist * hand_len_px
                            self.size_to_depth_c[label] = 0.9 * self.size_to_depth_c[label] + 0.1 * current_c
                        else:
                            final_dist = self.size_to_depth_c[label] / hand_len_px

                        if 0.05 < final_dist < 2.0: 
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [depth_x, cy], final_dist)
                            msg = Point(x=float(p3d[0]), y=float(p3d[1]), z=float(p3d[2]))
                            self.target_pub.publish(msg)
                        
                        self.mp_drawing.draw_landmarks(display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # 손 라벨링 색상 (L: 파란색, R: 빨간색)
                        text_color = (255, 150, 0) if label == "Left" else (0, 100, 255)
                        cv2.putText(display, f"{label}: {final_dist:.2f}m [{dist_source}]", (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # 프레임 및 상태 표시
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 30.0
            cv2.rectangle(display, (0, 0), (640, 40), (0, 0, 0), -1)
            cv2.putText(display, "Lightweight Multi Tracker (L:Blue, R:Red)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"FPS: {fps:.1f}", (540, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Unified Hand & Face', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
                
            # 강제 30FPS 휴식 (CPU 양보)
            sleep_time = (1.0 / 30.0) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def destroy_node(self):
        self.is_running = False
        self.vision_thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LightweightMultiTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__': main()
