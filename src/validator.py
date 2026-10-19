# src/validator.py
import numpy as np

class PoseValidator:
    def __init__(self):
        self.errors = []
        self.frame_count = 0
        self.warmup_frames = 30  # Skip the first 1 second (~30 frames) while positioning

    def compute_mpjpe(self, pred_joint, gt_joint):
        self.frame_count += 1
        
        # 3D Euclidean distance in millimeters
        error = np.linalg.norm(np.array(pred_joint) - np.array(gt_joint)) * 1000.0

        # Only accumulate stats after the warm-up period and filter out setup anomalies (> 250 mm)
        if self.frame_count > self.warmup_frames and error < 250.0:
            self.errors.append(error)

        return error

    def get_summary_statistics(self):
        if not self.errors:
            return 0.0, 0.0
        mean_error = np.mean(self.errors)
        std_error = np.std(self.errors)
        return float(mean_error), float(std_error)