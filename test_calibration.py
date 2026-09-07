import cv2
from vision.calibration import detect_aruco_marker

image = cv2.imread("calibration_marker.png")

if image is None:
    print("Could not load calibration_marker.png — check the file is in this folder.")
else:
    result = detect_aruco_marker(image, marker_size_mm=50.0)
    print(result)