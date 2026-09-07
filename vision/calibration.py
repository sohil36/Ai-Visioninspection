import cv2
import numpy as np


def detect_aruco_marker(image, marker_size_mm=80.0):
    """
    Detect an ArUco marker and calculate pixel-to-mm scale.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters
    )

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {
            "success": False,
            "message": "Calibration marker not detected"
        }

    # Use the first detected marker
    marker_corners = corners[0][0]

    # Calculate four side lengths in pixels
    side1 = cv2.norm(marker_corners[0], marker_corners[1])
    side2 = cv2.norm(marker_corners[1], marker_corners[2])
    side3 = cv2.norm(marker_corners[2], marker_corners[3])
    side4 = cv2.norm(marker_corners[3], marker_corners[0])

    average_pixel_size = (
        side1 + side2 + side3 + side4
    ) / 4.0

    mm_per_pixel = marker_size_mm / average_pixel_size

    return {
        "success": True,
        "marker_id": int(np.array(ids[0]).flatten()[0]),
        "marker_pixel_size": round(average_pixel_size, 2),
        "marker_size_mm": marker_size_mm,
        "mm_per_pixel": round(mm_per_pixel, 6),
        "marker_corners": marker_corners.tolist()
    }
