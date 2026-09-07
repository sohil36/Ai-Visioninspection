import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_id = 0
marker_size = 400  # pixels, the black/white pattern itself

marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

# Add a white border (quiet zone) around it — required for detection
border_size = 60
bordered = cv2.copyMakeBorder(
    marker_image,
    border_size, border_size, border_size, border_size,
    cv2.BORDER_CONSTANT,
    value=255
)

cv2.imwrite("calibration_marker.png", bordered)
print("Marker regenerated with white border:", bordered.shape)
