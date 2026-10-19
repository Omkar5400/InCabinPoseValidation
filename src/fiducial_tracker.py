# src/fiducial_tracker.py
import cv2
import numpy as np

class FiducialTracker:
    def __init__(self, camera_matrix, dist_coeffs, marker_length=0.05):
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self.marker_length = marker_length  # In meters
        
        # Initialize ArUco dictionary (DICT_4X4_50 is robust and fast)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        poses = {}

        if ids is not None:
            # 3D points of the marker corners in marker coordinate space
            obj_points = np.array([
                [-self.marker_length / 2,  self.marker_length / 2, 0],
                [ self.marker_length / 2,  self.marker_length / 2, 0],
                [ self.marker_length / 2, -self.marker_length / 2, 0],
                [-self.marker_length / 2, -self.marker_length / 2, 0]
            ], dtype=np.float32)

            for i, marker_id in enumerate(ids.flatten()):
                # SolvePnP for marker pose
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corners[i], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if success:
                    poses[int(marker_id)] = {
                        "tvec": tvec.flatten(),  # (x, y, z) in meters relative to camera
                        "rvec": rvec.flatten(),
                        "corners": corners[i]
                    }
        return poses