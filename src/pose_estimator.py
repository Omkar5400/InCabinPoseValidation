# src/pose_estimator.py
import cv2
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_name="yolov8n-pose.pt"):
        self.model = YOLO(model_name)

    def process_frame(self, frame, camera_matrix):
        results = self.model(frame, verbose=False)
        
        keypoints_dict = {}
        detected = False

        if len(results) > 0 and results[0].keypoints is not None:
            kp_data = results[0].keypoints
            # Check that at least one person was detected (shape is [num_persons, 17, 2])
            if kp_data.xy is not None and kp_data.xy.shape[0] > 0:
                detected = True
                coords_2d = kp_data.xy[0].cpu().numpy()  # (17, 2)
                conf = kp_data.conf[0].cpu().numpy() if kp_data.conf is not None else np.ones(17)

                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                cx = camera_matrix[0, 2]
                cy = camera_matrix[1, 2]

                # Lift 2D pixel coordinates to approximate metric camera frame (Z ~ 0.85m nominal distance)
                z_depth = 0.85 
                for idx, (pt, c) in enumerate(zip(coords_2d, conf)):
                    if c > 0.4:
                        u, v = pt[0], pt[1]
                        x_m = (u - cx) * z_depth / fx
                        y_m = (v - cy) * z_depth / fy
                        keypoints_dict[idx] = np.array([x_m, y_m, z_depth], dtype=np.float32)

        return results[0] if detected else None, keypoints_dict

    def draw_landmarks_on_frame(self, frame, result):
        if result is not None:
            return result.plot()
        return frame