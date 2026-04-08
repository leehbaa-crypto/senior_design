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

class UnifiedHandTracker(Node):
    def __init__(self):
        super().__init__('unified_hand_tracker')
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, # [통합 1] 양손 모두 인식
            model_complexity=0, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.5
        )

        self.pipeline = rs.pipeline()
        config = rs.config()
        # 해상도 640x480, 30FPS로 하드웨어 부하 최소화
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except Exception as e:
            self.get_logger().error(f"RealSense Error: {e}")
            sys.exit(1)

        # [핵심 통합 2] 양손의 크기가 다를 수 있으므로 캘리브레이션 상수를 좌/우 독립적으로 관리
        self.size_to_depth_c = {"Left": 40.0, "Right": 40.0} 
        
        self.is_running = True
        self.vision_thread = threading.Thread(target=self.camera_loop)
        self.vision_thread.start()
        self.get_logger().info("⚡ Unified Hybrid Tracker Started (2 Hands + Optimized FPS)")

    def camera_loop(self):
        while self.is_running and rclpy.ok():
            start_time = time.time()

            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame: 
                continue

            image = np.asanyarray(color_frame.get_data())
            display = cv2.flip(image, 1) # 직관성을 위한 거울 모드 (좌우 반전 유지)
            
            # MediaPipe 연산
            results = self.hands.process(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # [수정 포인트 1] 거울 모드에서 라벨 출력 보정
                    # MediaPipe의 원본 라벨을 그대로 가져옵니다. 
                    # (만약 테스트 해보시고 여전히 반대라면 label = "Right" if original_label == "Left" else "Left" 로 변경해주세요)
                    label = results.multi_handedness[idx].classification[0].label 
                    
                    cx = int(hand_landmarks.landmark[9].x * 640)
                    cy = int(hand_landmarks.landmark[9].y * 480)
                    wx = int(hand_landmarks.landmark[0].x * 640)
                    wy = int(hand_landmarks.landmark[0].y * 480)

                    if 20 < cx < 620 and 20 < cy < 460: 
                        hand_len_px = max(5.0, np.sqrt((cx - wx)**2 + (cy - wy)**2))
                        
                        # Depth x좌표도 거울 모드에 맞게 반전하여 추출
                        depth_x = min(639, max(0, 639 - cx))
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

                        # ROS 전송
                        if 0.05 < final_dist < 2.0: 
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [depth_x, cy], final_dist)
                            msg = Point(x=float(p3d[0]), y=float(p3d[1]), z=float(p3d[2]))
                            self.target_pub.publish(msg)
                        
                        # [수정 포인트 2] UI 시각화를 왼손/오른손 확실하게 다르게 출력
                        self.mp_drawing.draw_landmarks(display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # 왼손은 파란색, 오른손은 빨간색으로 글자 색상 분리 (OpenCV는 BGR 순서)
                        if label == "Left":
                            text_color = (255, 150, 0) # 파란색 계열
                        else:
                            text_color = (0, 100, 255) # 빨간색 계열
                            
                        # 손 위에 라벨(좌/우), 거리, 출처 표시
                        cv2.putText(display, f"{label}: {final_dist:.2f}m [{dist_source}]", (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 30.0
            cv2.rectangle(display, (0, 0), (640, 40), (0, 0, 0), -1)
            cv2.putText(display, "Unified Tracker (L:Blue, R:Red)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"FPS: {fps:.1f}", (540, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Unified Tracker', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
                
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
    node = UnifiedHandTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__': main()
