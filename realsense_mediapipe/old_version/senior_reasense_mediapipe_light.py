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

class FastHandTracker(Node):
    def __init__(self):
        super().__init__('fast_hand_tracker')
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, # [최적화 1] 회피를 위한 추적이라면 일단 한 손만 집중해서 연산량 절반으로 단축
            model_complexity=0, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.5
        )

        self.pipeline = rs.pipeline()
        config = rs.config()
        # 해상도를 640x480으로 유지하되 30FPS로 설정
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except Exception as e:
            self.get_logger().error(f"RealSense Error: {e}")
            sys.exit(1)

        self.size_to_depth_c = 40.0 
        self.is_running = True
        self.vision_thread = threading.Thread(target=self.camera_loop)
        self.vision_thread.start()
        self.get_logger().info("⚡ Lightweight Hybrid Tracker Started")

    def camera_loop(self):
        while self.is_running and rclpy.ok():
            start_time = time.time()

            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame: 
                continue

            image = np.asanyarray(color_frame.get_data())
            display = cv2.flip(image, 1) # 좌우 반전
            
            # MediaPipe 연산
            results = self.hands.process(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                cx = int(hand_landmarks.landmark[9].x * 640)
                cy = int(hand_landmarks.landmark[9].y * 480)
                wx = int(hand_landmarks.landmark[0].x * 640)
                wy = int(hand_landmarks.landmark[0].y * 480)

                if 20 < cx < 620 and 20 < cy < 460: # [최적화 2] 가장자리 노이즈 무시
                    hand_len_px = max(5.0, np.sqrt((cx - wx)**2 + (cy - wy)**2))
                    
                    depth_x = min(639, max(0, 639 - cx))
                    raw_dist = depth_frame.get_distance(depth_x, cy)
                    
                    final_dist = 0.0
                    dist_source = "Vision"

                    # [최적화 3] 캘리브레이션 안정성 확보: 손 크기가 일정 수준 이상 보일 때만 학습
                    if 0.35 < raw_dist < 1.2 and hand_len_px > 40:
                        final_dist = raw_dist
                        dist_source = "LiDAR"
                        current_c = raw_dist * hand_len_px
                        self.size_to_depth_c = 0.9 * self.size_to_depth_c + 0.1 * current_c
                    else:
                        final_dist = self.size_to_depth_c / hand_len_px

                    # ROS 전송
                    if 0.05 < final_dist < 2.0: 
                        p3d = rs.rs2_deproject_pixel_to_point(self.intr, [depth_x, cy], final_dist)
                        msg = Point(x=float(p3d[0]), y=float(p3d[1]), z=float(p3d[2]))
                        self.target_pub.publish(msg)
                    
                    # 그리기 (필요 최소한)
                    self.mp_drawing.draw_landmarks(display, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    color = (0, 0, 255) if final_dist < 0.4 else (0, 255, 255)
                    cv2.putText(display, f"{final_dist:.2f}m [{dist_source}]", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 프레임 속도 맞추기 (최대 30FPS)
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 30.0
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Optimized Tracker', display)
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
    node = FastHandTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__': main()
