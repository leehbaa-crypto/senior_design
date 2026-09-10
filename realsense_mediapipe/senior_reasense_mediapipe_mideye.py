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

class UltraLightMultiTracker(Node):
    def __init__(self):
        super().__init__('ultra_light_tracker')
        self.left_pub = self.create_publisher(Point, '/left_hand', 10)
        self.right_pub = self.create_publisher(Point, '/right_hand', 10)
        # [추가] 얼굴 좌표 전송용 퍼블리셔
        self.face_pub = self.create_publisher(Point, '/face_pose', 10)

        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

    def camera_loop(self):
        while self.is_running and rclpy.ok():
            start_time = time.time()
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue

            image = np.asanyarray(color_frame.get_data())
            display = cv2.flip(image, 1) 
            rgb_display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            
            hand_results = self.hands.process(rgb_display)
            face_results = self.face_mesh.process(rgb_display)
            
            # [수정] 얼굴 인식 및 3D 좌표 추출 로직
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # 윤곽선 그리기
                    self.mp_drawing.draw_landmarks(display, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        None, self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                    # 168번 랜드마크(미간) 좌표 추출
                    nx, ny = int(face_landmarks.landmark[168].x * 640), int(face_landmarks.landmark[168].y * 480)
                    if 20 < nx < 620 and 20 < ny < 460:
                        depth_nx = min(639, max(0, 639 - nx)) # 거울 모드 보정
                        face_dist = depth_frame.get_distance(depth_nx, ny)
                        
                        if 0.2 < face_dist < 2.5: # 얼굴 인식 유효 거리
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [depth_nx, ny], face_dist)
                            msg = Point(x=float(p3d[0]), y=float(p3d[1]), z=float(p3d[2]))
                            self.face_pub.publish(msg)
                            # 미간에 노란 점 표시
                            cv2.circle(display, (nx, ny), 6, (0, 255, 255), -1)

            if hand_results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    label = hand_results.multi_handedness[idx].classification[0].label 
                    
                    cx, cy = int(hand_landmarks.landmark[9].x * 640), int(hand_landmarks.landmark[9].y * 480)
                    wx, wy = int(hand_landmarks.landmark[0].x * 640), int(hand_landmarks.landmark[0].y * 480)

                    if 20 < cx < 620 and 20 < cy < 460: 
                        hand_len_px = max(5.0, np.sqrt((cx - wx)**2 + (cy - wy)**2))
                        depth_x = min(639, max(0, 639 - cx)) 
                        raw_dist = depth_frame.get_distance(depth_x, cy)
                        
                        if 0.35 < raw_dist < 1.2 and hand_len_px > 40:
                            final_dist = raw_dist
                            self.size_to_depth_c[label] = 0.9 * self.size_to_depth_c[label] + 0.1 * (raw_dist * hand_len_px)
                        else:
                            final_dist = self.size_to_depth_c[label] / hand_len_px

                        if 0.05 < final_dist < 2.0: 
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [depth_x, cy], final_dist)
                            msg = Point(x=float(p3d[0]), y=float(p3d[1]), z=float(p3d[2]))
                            
                            if label == "Left": self.left_pub.publish(msg)
                            else: self.right_pub.publish(msg)
                        
                        self.mp_drawing.draw_landmarks(display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        text_color = (0, 0, 255) if label == "Left" else (0, 255, 0)
                        cv2.putText(display, f"{label}: {final_dist:.2f}m", (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 30.0
            cv2.putText(display, f"FPS: {fps:.1f} | L:Red, R:Green, Face:Yellow", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow('Ultra Light Tracker', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(max(0, (1.0 / 30.0) - (time.time() - start_time)))

    def destroy_node(self):
        self.is_running = False
        self.vision_thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args); node = UltraLightMultiTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__': main()
