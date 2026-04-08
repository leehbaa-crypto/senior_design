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

class MultiTracker(Node):
    def __init__(self):
        super().__init__('multi_tracker')
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)

        # MediaPipe 솔루션 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh # 얼굴 메쉬 추가
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 손 추적기 설정
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

        # 얼굴 메쉬 추적기 설정 (세부 인식 모드)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # 눈동자, 입술 등 더 세밀한 인식 활성화
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # RealSense 초기화
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

        self.is_running = True
        self.vision_thread = threading.Thread(target=self.camera_loop)
        self.vision_thread.start()
        self.get_logger().info("✅ Multi Tracker Started (Hand + Face Mesh)")

    def camera_loop(self):
        while self.is_running and rclpy.ok():
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue

            image = np.asanyarray(color_frame.get_data())
            # 처리를 위해 RGB로 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. 손 추적 실행
            hand_results = self.hands.process(rgb_image)
            # 2. 얼굴 메쉬 추적 실행
            face_results = self.face_mesh.process(rgb_image)

            # 시각화용 복사본
            display = image.copy()

            # --- 손 시각화 및 좌표 전송 ---
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 손바닥 중심(9번) 좌표 추출 및 ROS 전송
                    cx = int(hand_landmarks.landmark[9].x * 640)
                    cy = int(hand_landmarks.landmark[9].y * 480)
                    if 0 <= cx < 640 and 0 <= cy < 480:
                        dist = depth_frame.get_distance(cx, cy)
                        if 0.1 < dist < 2.0:
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], dist)
                            msg = Point(x=float(p3d[0]), y=float(p3d[1]), z=float(p3d[2]))
                            self.target_pub.publish(msg)

            # --- 얼굴 메쉬 시각화 ---
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # 얼굴 전체 그물망(Mesh) 그리기
                    self.mp_drawing.draw_landmarks(
                        image=display,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    # 눈, 눈썹, 입술 윤곽선 그리기
                    self.mp_drawing.draw_landmarks(
                        image=display,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                    # 눈동자(Iris) 그리기
                    self.mp_drawing.draw_landmarks(
                        image=display,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            cv2.imshow('Hand & Face Mesh Tracker', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    def destroy_node(self):
        self.is_running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MultiTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__':
    main()
