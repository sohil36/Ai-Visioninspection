import cv2
import numpy as np


def measure_rectangular_object(image, mm_per_pixel, marker_corners):
    """
    Detects the largest rectangular object in the image (excluding the
    calibration marker itself) and measures its width and height in mm.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection + threshold to find object outlines
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "success": False,
            "message": "No object detected in frame"
        }

    # Build a rough bounding area for the marker so we can exclude it
    marker_x, marker_y, marker_w, marker_h = cv2.boundingRect(
        np.array(marker_corners, dtype=np.int32)
    )

    best_contour = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue  # skip tiny noise contours

        x, y, w, h = cv2.boundingRect(c)

        # Skip contour if it overlaps heavily with the marker region
        overlap_x = max(0, min(x + w, marker_x + marker_w) - max(x, marker_x))
        overlap_y = max(0, min(y + h, marker_y + marker_h) - max(y, marker_y))
        overlap_area = overlap_x * overlap_y

        if overlap_area > 0.5 * (w * h):
            continue  # this contour is mostly the marker itself, skip

        if area > best_area:
            best_area = area
            best_contour = c

    if best_contour is None:
        return {
            "success": False,
            "message": "Could not isolate object from marker"
        }

    rect = cv2.minAreaRect(best_contour)
    (cx, cy), (w_px, h_px), angle = rect

    width_mm = round(w_px * mm_per_pixel, 2)
    height_mm = round(h_px * mm_per_pixel, 2)

    return {
        "success": True,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "width_px": round(w_px, 2),
        "height_px": round(h_px, 2),
        "rotation_deg": round(angle, 2)
    }
