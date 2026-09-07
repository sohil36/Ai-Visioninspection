import cv2
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


def evaluate_pass_fail(measurement_result):
    if not measurement_result.get("success"):
        return {"result": "FAIL", "reason": "Measurement failed"}

    w = measurement_result.get("width_mm", 0)
    h = measurement_result.get("height_mm", 0)
    identification = measurement_result.get("identification", {})
    component_type = identification.get("component_type", "Unknown")

    if w <= 0 or h <= 0:
        return {"result": "FAIL", "reason": "Invalid dimensions (<=0)"}

    if "Unknown" in component_type:
        return {"result": "FAIL", "reason": "Component could not be confidently identified"}

    return {"result": "PASS", "reason": "Dimensions valid and component identified"}


def generate_pdf_report(image, calibration_result, measurement_result, output_dir="reports"):
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{output_dir}/capture_{timestamp}.jpg"
    pdf_path = f"{output_dir}/report_{timestamp}.pdf"

    cv2.imwrite(image_path, image)

    pass_fail = evaluate_pass_fail(measurement_result)
    identification = measurement_result.get("identification", {})

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    y = height - 30 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "AI Vision Inspection Report")

    y -= 10 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        y -= 70 * mm
        c.drawImage(image_path, 20 * mm, y, width=80 * mm, height=60 * mm, preserveAspectRatio=True)
    except Exception:
        pass

    y -= 10 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Identification")
    c.setFont("Helvetica", 10)
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Component Type: {identification.get('component_type', 'N/A')} (AI-estimated, geometric heuristic)")
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Reason: {identification.get('reason', 'N/A')}")

    y -= 10 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Measurements (Directly measured, camera + ArUco calibration)")
    c.setFont("Helvetica", 10)
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Width: {measurement_result.get('width_mm', 'N/A')} mm")
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Height: {measurement_result.get('height_mm', 'N/A')} mm")
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Has internal hole: {measurement_result.get('has_internal_hole', 'N/A')}")

    y -= 10 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Calibration Reference")
    c.setFont("Helvetica", 10)
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Marker ID: {calibration_result.get('marker_id', 'N/A')} (ArUco DICT_4X4_50)")
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Scale: {calibration_result.get('mm_per_pixel', 'N/A')} mm/pixel")

    y -= 12 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, f"Result: {pass_fail['result']}")
    c.setFont("Helvetica", 10)
    y -= 6 * mm
    c.drawString(20 * mm, y, f"Reason: {pass_fail['reason']}")

    c.save()

    return {
        "pdf_path": pdf_path,
        "pass_fail": pass_fail
    }
