import cv2
import numpy as np


def classify_component(contour):
    """
    Classifies a detected contour as one of: Plate, Washer, Shaft,
    Hex Bolt Head, or Unknown — using classical geometric heuristics.
    Returns the identification along with the confidence signals used,
    so results are explainable rather than a black-box guess.
    """

    area = cv2.contourArea(contour)
    if area <= 0:
        return {
            "component_type": "Unknown",
            "reason": "Contour area is zero or invalid"
        }

    perimeter = cv2.arcLength(contour, True)

    # Approximate the contour to a polygon to count straight edges
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Circularity: 1.0 = perfect circle, lower = more irregular/angular
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    # Rectangle fit quality: how much of the min-area rect is filled by the contour
    rect = cv2.minAreaRect(contour)
    (rect_w, rect_h) = rect[1]
    rect_area = rect_w * rect_h
    rectangularity = area / rect_area if rect_area > 0 else 0

    result = {
        "num_vertices": int(num_vertices),
        "circularity": float(round(circularity, 3)),
        "rectangularity": float(round(rectangularity, 3))
    }

    if circularity > 0.80:
        result["component_type"] = "Round component (shaft, washer, or disc)"
        result["reason"] = f"High circularity ({circularity:.2f}) indicates a round profile"

    elif num_vertices == 6 and circularity > 0.55:
        result["component_type"] = "Hex Bolt / Hex Nut"
        result["reason"] = f"6 straight edges detected with moderate circularity ({circularity:.2f}), consistent with a hex head"

    elif num_vertices == 4 and rectangularity > 0.75:
        result["component_type"] = "Plate / Bracket"
        result["reason"] = f"4 corners detected, contour fills {rectangularity:.0%} of its bounding rectangle"

    else:
        result["component_type"] = "Unknown / Irregular"
        result["reason"] = f"{num_vertices} vertices, circularity {circularity:.2f}, rectangularity {rectangularity:.2f} — does not clearly match known profiles"

    return result


def detect_internal_hole(image, outer_contour_index, contours, hierarchy):
    """
    Checks whether a given contour has a child contour (a hole) inside it.
    Used to distinguish a washer (has a hole) from a solid disc/shaft.
    """

    if hierarchy is None:
        return False

    # hierarchy[0][i] = [next, previous, first_child, parent]
    first_child = hierarchy[0][outer_contour_index][2]

    return bool(first_child != -1)
