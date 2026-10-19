import cv2
import yaml
import numpy as np
from src.fiducial_tracker import FiducialTracker
from src.pose_estimator import PoseEstimator
from src.ekf_refinement import JointEKF, enforce_bone_length
from src.validator import PoseValidator

def main():
    # 1. Load or fallback camera calibration
    try:
        with open("config/camera_calibration.yaml", "r") as f:
            calib = yaml.safe_load(f)
            if calib and "camera_matrix" in calib:
                K = np.array(calib["camera_matrix"], dtype=np.float32)
                D = np.array(calib["dist_coeff"], dtype=np.float32)
            else:
                raise ValueError
    except Exception:
        print("Calibration file empty/not found. Using default intrinsic matrix.")
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        D = np.zeros(5, dtype=np.float32)

    # 2. Initialize tracking and estimation modules
    tracker = FiducialTracker(K, D, marker_length=0.06)
    estimator = PoseEstimator()
    wrist_filter = JointEKF()
    validator = PoseValidator()

    # 3. Robust AVFoundation camera capture
    cap = None
    for camera_idx in [0, 1]:
        temp_cap = cv2.VideoCapture(camera_idx, cv2.CAP_AVFOUNDATION)
        if temp_cap.isOpened():
            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                print(f"Connected to camera index {camera_idx}")
                cap = temp_cap
                break
            temp_cap.release()

    if cap is None or not cap.isOpened():
        print("Error: Could not open camera device. Check macOS permissions.")
        return

    print("Running in-cabin pose validation pipeline. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # --- A. ArUco Marker Detection (Ground Truth) ---
        fiducials = tracker.detect_markers(frame)

        # --- B. Human Pose Estimation (YOLOv8-Pose) ---
        result, keypoints = estimator.process_frame(frame, K)

        # Draw the pose landmarks on the frame
        if result is not None:
            frame = estimator.draw_landmarks_on_frame(frame, result)

            # YOLOv8 keypoint index 10 is Right Wrist
            if 10 in keypoints:
                raw_wrist = keypoints[10]
                wrist_filter.predict()
                filtered_wrist = wrist_filter.update(raw_wrist)

                # Draw a distinct green circle on the filtered wrist point
                u_pix = int((filtered_wrist[0] * K[0, 0] / filtered_wrist[2]) + K[0, 2])
                v_pix = int((filtered_wrist[1] * K[1, 1] / filtered_wrist[2]) + K[1, 2])
                if 0 <= u_pix < frame.shape[1] and 0 <= v_pix < frame.shape[0]:
                    cv2.circle(frame, (u_pix, v_pix), 8, (0, 255, 0), -1)
                    cv2.putText(frame, "Refined Wrist (EKF)", (u_pix + 12, v_pix),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # If ArUco tag ID 1 is detected, evaluate MPJPE
                if 1 in fiducials:
                    gt_wrist = fiducials[1]["tvec"]
                    mpjpe = validator.compute_mpjpe(filtered_wrist, gt_wrist)
                    cv2.putText(frame, f"Wrist MPJPE: {mpjpe:.1f} mm", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- C. Draw ArUco Ground Truth Boundaries ---
        if fiducials:
            for marker_id, data in fiducials.items():
                corners = np.int32(data["corners"]).reshape(-1, 2)
                cv2.polylines(frame, [corners], True, (0, 0, 255), 2)
                cv2.putText(frame, f"ArUco ID: {marker_id}", (corners[0][0], corners[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- D. Live Status Overlay ---
        pose_status = "DETECTED" if (result is not None) else "SEARCHING"
        color = (0, 255, 0) if (result is not None) else (0, 0, 255)
        cv2.putText(frame, f"Pose Model: {pose_status}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("In-Cabin Pose & Fiducial Validation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    mean_err, std_err = validator.get_summary_statistics()
    print(f"\nValidation Complete.")
    print(f"Mean Per Joint Position Error (MPJPE): {mean_err:.2f} mm (+/- {std_err:.2f} mm)")

if __name__ == "__main__":
    main()