from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import numpy as np

from vision.calibration import detect_aruco_marker
from vision.measurement import measure_rectangular_object
from vision.report import generate_pdf_report

app = FastAPI(title="AI Vision Inspection")

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

app.mount(
    "/reports",
    StaticFiles(directory=BASE_DIR / "reports"),
    name="reports"
)


@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "application": "AI Vision Inspection"
    }


@app.post("/calibrate")
async def calibrate(file: UploadFile = File(...)):

    data = await file.read()

    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {"success": False, "message": "Invalid image"}

    result = detect_aruco_marker(image, marker_size_mm=80.0)

    if not result["success"]:
        return result

    return {
        "success": True,
        "message": "Calibration successful",
        "marker_id": result["marker_id"],
        "marker_pixel_size": result["marker_pixel_size"],
        "mm_per_pixel": result["mm_per_pixel"]
    }


@app.post("/measure")
async def measure(file: UploadFile = File(...)):

    data = await file.read()

    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {"success": False, "message": "Invalid image"}

    calibration_result = detect_aruco_marker(image, marker_size_mm=80.0)

    if not calibration_result["success"]:
        return {
            "success": False,
            "message": "Calibration failed: " + calibration_result["message"]
        }

    measurement_result = measure_rectangular_object(
        image,
        calibration_result["mm_per_pixel"],
        calibration_result["marker_corners"]
    )

    if not measurement_result["success"]:
        return {
            "success": False,
            "message": "Measurement failed: " + measurement_result["message"]
        }

    return {
        "success": True,
        "message": "Measurement successful",
        "calibration": {
            "marker_id": calibration_result["marker_id"],
            "mm_per_pixel": calibration_result["mm_per_pixel"]
        },
        "measurement": measurement_result
    }


@app.post("/report")
async def report(file: UploadFile = File(...)):

    data = await file.read()

    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {"success": False, "message": "Invalid image"}

    calibration_result = detect_aruco_marker(image, marker_size_mm=80.0)

    if not calibration_result["success"]:
        return {
            "success": False,
            "message": "Calibration failed: " + calibration_result["message"]
        }

    measurement_result = measure_rectangular_object(
        image,
        calibration_result["mm_per_pixel"],
        calibration_result["marker_corners"]
    )

    if not measurement_result["success"]:
        return {
            "success": False,
            "message": "Measurement failed: " + measurement_result["message"]
        }

    report_result = generate_pdf_report(image, calibration_result, measurement_result)

    return {
        "success": True,
        "pdf_url": "/" + report_result["pdf_path"].replace("\\", "/"),
        "pass_fail": report_result["pass_fail"]
    }
