# src/calibration.py
import cv2
import numpy as np
import yaml

def calibrate_camera(square_size=0.025, pattern_size=(9, 6)):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    cap = cv2.VideoCapture(0)
    print("Capturing frames for calibration. Press 'c' to capture, 'q' when done (need 15-20 frames).")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        display_frame = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, pattern_size, corners, ret_corners)

        cv2.imshow("Calibration Capture", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret_corners:
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
            print(f"Captured frame {count}")
        elif key == ord('q') and count >= 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    calib_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist()
    }
    with open("config/camera_calibration.yaml", "w") as f:
        yaml.dump(calib_data, f)
    print("Calibration saved successfully to config/camera_calibration.yaml")
    return mtx, dist

if __name__ == "__main__":
    calibrate_camera()