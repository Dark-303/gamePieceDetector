import cv2
from ultralytics import YOLO
import numpy as np

# Load your trained model
model = YOLO("runs/train/coral_modelVER1/coral_modelVER1.3/weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

THRESHOLD = True
CONFIDENCE_THRESHOLD = 0.2  # 20% confidence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)[0]  # results[0] is for this frame

    # Filter boxes by confidence
    if results.boxes is not None and len(results.boxes) > 0:
        mask = results.boxes.conf >= CONFIDENCE_THRESHOLD
        results.boxes = results.boxes[mask]  # only keep boxes >= threshold

    # Visualize
    annotated_frame = results.plot()

    cv2.imshow("Coral Detector Live", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()