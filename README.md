# AI Vision Inspection & Measurement

A web-based AI-assisted engineering inspection tool that uses a smartphone
camera to identify, measure, and inspect mechanical components — built for
the Nebula KnowLab take-home assignment.

## Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Computer Vision:** OpenCV (classical CV — ArUco detection, contour
  analysis, geometric heuristics)
- **Frontend:** Plain HTML/JavaScript, using the browser's `getUserMedia`
  camera API (no framework, to keep the phone-camera pipeline simple and
  dependency-free)
- **Hosting for phone access:** ngrok (HTTPS tunnel), since browser camera
  access requires a secure context and the phone and dev machine are not
  on a setup with local HTTPS certs

## Why classical CV instead of a trained AI model

For calibration and measurement, this project deliberately uses **classical,
deterministic computer vision** (ArUco marker detection, contour analysis,
geometric shape heuristics) rather than a trained neural network. Reasoning:

- **Interpretability:** every measurement and classification decision can be
  traced back to a concrete geometric signal (vertex count, circularity,
  rectangularity, hole presence) — this matters for an *engineering
  inspection* tool, where a "why" behind a PASS/FAIL matters as much as the
  result itself.
- **No training data available** in the assignment's timeframe — a
  from-scratch CNN would be undertrained and less reliable than well-reasoned
  classical CV on this small, well-defined problem (a handful of standard
  mechanical shapes).
- **Repeatability:** classical CV gives the same output for the same input
  every time, which is important for a measurement tool where consistency is
  part of correctness.

AI/ML would become the better tool for tasks classical CV genuinely can't do
well within this scope — e.g. robust segmentation in cluttered scenes, or
fine surface-defect detection — noted as limitations below rather than
solved with an undertrained model.

## Calibration approach

- A printed [ArUco marker](https://docs.opencv.org/) (`DICT_4X4_50`, ID 0) of
  a known physical size is placed in the same frame as the component.
- The marker's real-world size is measured once by hand with a ruler after
  printing/display, since printers and screens do not reliably reproduce an
  exact requested size — measuring the ground truth is more trustworthy than
  trusting the print/display pipeline.
- OpenCV's ArUco detector finds the marker's four corners in the image;
  the average of its four side lengths (in pixels) is divided into the
  known real-world size (mm) to get a `mm_per_pixel` scale factor.
- This scale factor is then applied to any other detected object's
  pixel dimensions in the same frame to get real-world measurements.

## What this system measures reliably

- Overall width/height of a single, isolated, flat object against a plain,
  high-contrast background, photographed roughly perpendicular to the
  camera (minimal perspective tilt).
- Whether an object has a clear circular outer boundary and an internal hole
  (used to distinguish washer-like parts from solid discs/shafts).
- Whether an object's outline has 4 sharp corners (rectangular parts) vs.
  6 corners (hex heads) vs. a highly circular boundary.

## Known limitations (what this system does NOT measure reliably)

- **Cluttered or low-contrast scenes:** if a hand, background texture, or
  other objects are also in frame, Canny-edge-based contour detection can
  merge them into the component's silhouette, producing an incorrect shape
  and a wrong size/classification. Observed directly during testing: a
  small round bottle cap held in-hand was measured as a much larger
  rectangular "Plate/Bracket" (126mm × 38mm) because the hand's outline
  fused with the cap's outline into one contour. **Mitigation used:**
  photograph components on a flat, plain, contrasting background with no
  hand or clutter in frame.
- **Camera perspective / tilt:** the pixel-to-mm scale from the marker is
  only valid for objects lying in the same plane as the marker. If the
  component is tilted relative to the marker or camera, measured dimensions
  will be foreshortened and inaccurate. No perspective correction (e.g.
  homography warp) is currently applied.
- **Lens distortion:** smartphone wide-angle lenses introduce barrel
  distortion, especially near frame edges. This is not currently corrected
  per-device, so measurements of objects near the edge of frame carry more
  error than objects near the center.
- **Fine features:** thread pitch, small chamfers, and sub-millimeter
  defects are below the reliable resolution of a single phone photo at
  typical working distances, and are not attempted.
- **Single-view depth/height:** true 3D dimensions (e.g. component
  thickness when photographed from directly above) cannot be measured from
  a single 2D image; only the two in-plane dimensions are captured
  reliably.

## Error / uncertainty model

*(to be expanded — tolerances and confidence scoring)*

## Status

- [x] ArUco-based calibration
- [x] Rectangular/circular object measurement
- [x] Shape-based component identification
- [ ] PASS/FAIL logic
- [ ] PDF report generation
- [ ] Defect detection

