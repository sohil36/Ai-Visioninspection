import cv2

def detect_aruco_marker(image, marker_size_mm=50.0):
    """
    Detects a 4x4 ArUco marker with optimized detection parameters.
    """
    if image is None:
        return {"success": False, "message": "Invalid or empty image provided"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load 4x4 ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Tune detector parameters for low contrast, glare, and blur
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.polygonalApproxAccuracyRate = 0.05

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return {
            "success": False,
            "message": "Calibration marker not detected. Move closer and avoid glare."
        }

    # Extract corners for the first detected marker
    marker_corners = corners[0][0]

    # Calculate average side length in pixels
    side1 = cv2.norm(marker_corners[0], marker_corners[1])
    side2 = cv2.norm(marker_corners[1], marker_corners[2])
    side3 = cv2.norm(marker_corners[2], marker_corners[3])
    side4 = cv2.norm(marker_corners[3], marker_corners[0])

    average_pixel_size = (side1 + side2 + side3 + side4) / 4.0

    if average_pixel_size == 0:
        return {"success": False, "message": "Invalid marker dimensions detected"}

    mm_per_pixel = marker_size_mm / average_pixel_size

    return {
        "success": True,
        "marker_id": int(ids[0][0]),
        "marker_pixel_size": round(average_pixel_size, 2),
        "marker_size_mm": marker_size_mm,
        "mm_per_pixel": round(mm_per_pixel, 6),
        "marker_corners": marker_corners
    }
