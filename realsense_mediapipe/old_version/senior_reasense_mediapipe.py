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

class HandDistanceTracker(Node):
    def __init__(self):
        super().__init__('hand_distance_tracker')
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            model_complexity=0, 
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

        # [추가됨] 카메라 크기 비례 기반 거리 추정을 위한 상수 초기화
        # (임의의 초기값이며, 뎁스 센서가 켜지면 자동으로 사용자의 손 크기에 맞춰 보정됩니다)
        self.size_to_depth_c = 40.0 

        self.is_running = True
        self.vision_thread = threading.Thread(target=self.camera_loop)
        self.vision_thread.start()
        self.get_logger().info("✅ Hybrid Hand Distance Tracker Started (LiDAR + Vision Size)")

    def camera_loop(self):
        prev_time = time.time()
        
        while self.is_running and rclpy.ok():
            curr_time = time.time()
            if curr_time - prev_time < 0.033:
                time.sleep(0.01)
                continue
            prev_time = curr_time

            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame: 
                continue

            image = np.asanyarray(color_frame.get_data())
            display = cv2.flip(image, 1)
            results = self.hands.process(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    original_label = results.multi_handedness[idx].classification[0].label 
                    label = "Right" if original_label == "Left" else "Left"
                    
                    # 기준점 1: 손바닥 중앙 (중지 밑부분, 9번 랜드마크)
                    cx = int(hand_landmarks.landmark[9].x * 640)
                    cy = int(hand_landmarks.landmark[9].y * 480)
                    
                    # 기준점 2: 손목 (0번 랜드마크)
                    wx = int(hand_landmarks.landmark[0].x * 640)
                    wy = int(hand_landmarks.landmark[0].y * 480)

                    if 0 <= cx < 640 and 0 <= cy < 480:
                        # [핵심 로직] 화면상 손목~중지 관절 사이의 픽셀 길이 계산
                        hand_len_px = max(1.0, np.sqrt((cx - wx)**2 + (cy - wy)**2))
                        
                        # RealSense 라이다 원본 거리 추출
                        depth_x = min(639, max(0, 639 - cx))
                        raw_dist = depth_frame.get_distance(depth_x, cy)
                        
                        dist_source = ""
                        final_dist = 0.0

                        # L515 센서가 신뢰할 수 있는 구간 (예: 0.35m ~ 1.5m)
                        if 0.35 < raw_dist < 1.5:
                            final_dist = raw_dist
                            dist_source = "LiDAR"
                            # 신뢰할 수 있을 때 기준 상수를 지속적으로 학습 (Moving Average)
                            # 상수 = 실제 거리 * 화면상 픽셀 길이
                            current_c = raw_dist * hand_len_px
                            self.size_to_depth_c = 0.9 * self.size_to_depth_c + 0.1 * current_c
                        else:
                            # 거리가 너무 가깝거나(0.35m 이하) 너무 멀어 센서가 튈 때 -> 비전 기반 추정 폴백
                            final_dist = self.size_to_depth_c / hand_len_px
                            dist_source = "Vision-Est"

                        # 3D 좌표 변환 및 전송 (추정된 거리라도 카메라는 그 거리에 있다고 믿고 3D 좌표 생성)
                        if 0.05 < final_dist < 2.0: 
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [depth_x, cy], final_dist)
                            
                            msg = Point()
                            msg.x = float(p3d[0])
                            msg.y = float(p3d[1])
                            msg.z = float(p3d[2])
                            self.target_pub.publish(msg)
                        
                        self.mp_drawing.draw_landmarks(display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                        
                        # 화면에 거리와 출처 표시 (LiDAR 센서 값인지 추정 값인지)
                        text = f"{label}: {final_dist:.2f}m [{dist_source}]"
                        color = (0, 0, 255) if final_dist < 0.4 else (0, 255, 255)
                        cv2.putText(display, text, (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(display, (0, 0), (640, 40), (0, 0, 0), -1)
            cv2.putText(display, "Hybrid Tracker (LiDAR + Vision)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"FPS: {1.0/(time.time()-curr_time+0.001):.1f}", (540, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Hand Distance & Position', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    def destroy_node(self):
        self.is_running = False
        self.vision_thread.join()
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandDistanceTracker()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally: 
        node.destroy_node()

if __name__ == '__main__': 
    main()
