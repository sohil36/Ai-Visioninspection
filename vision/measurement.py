import cv2
import numpy as np

from vision.identification import classify_component, detect_internal_hole


def measure_rectangular_object(image, mm_per_pixel, marker_corners):
    """
    Detects the largest object in the image (excluding the calibration
    marker itself), measures its bounding dimensions in mm, and
    classifies its likely component type using geometric heuristics.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "success": False,
            "message": "No object detected in frame"
        }

    marker_x, marker_y, marker_w, marker_h = cv2.boundingRect(
        np.array(marker_corners, dtype=np.int32)
    )

    best_contour = None
    best_index = -1
    best_area = 0

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        overlap_x = max(0, min(x + w, marker_x + marker_w) - max(x, marker_x))
        overlap_y = max(0, min(y + h, marker_y + marker_h) - max(y, marker_y))
        overlap_area = overlap_x * overlap_y

        if overlap_area > 0.5 * (w * h):
            continue

        if area > best_area:
            best_area = area
            best_contour = c
            best_index = i

    if best_contour is None:
        return {
            "success": False,
            "message": "Could not isolate object from marker"
        }

    rect = cv2.minAreaRect(best_contour)
    (cx, cy), (w_px, h_px), angle = rect

    width_mm = round(w_px * mm_per_pixel, 2)
    height_mm = round(h_px * mm_per_pixel, 2)

    identification = classify_component(best_contour)
    has_hole = detect_internal_hole(image, best_index, contours, hierarchy)

    # Refine identification if a hole was found (distinguishes washer from disc)
    if has_hole and "Round component" in identification["component_type"]:
        identification["component_type"] = "Washer"
        identification["reason"] += " — internal hole detected, confirming washer profile"
    elif not has_hole and "Round component" in identification["component_type"]:
        identification["component_type"] = "Shaft / Disc (solid, no hole)"

    return {
        "success": True,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "width_px": round(w_px, 2),
        "height_px": round(h_px, 2),
        "rotation_deg": round(angle, 2),
        "identification": identification,
        "has_internal_hole": has_hole
    }
