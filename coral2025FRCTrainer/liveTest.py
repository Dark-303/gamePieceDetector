import cv2
from ultralytics import YOLO

# Model path
model = YOLO("versions/coral_modelVER1.3/coral_modelVER1.3.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

THRESHOLD = True
CONFIDENCE_THRESHOLD = 0.60  # Adjust based on model performance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)[0]

    # Filter boxes by confidence threshold
    if THRESHOLD:
        if results.boxes is not None and len(results.boxes) > 0:
            mask = results.boxes.conf >= CONFIDENCE_THRESHOLD
            results.boxes = results.boxes[mask]  # Only keep boxes >= threshold

    # Visualize
    annotated_frame = results.plot()

    # Display the resulting frame
    cv2.imshow("Coral Detector Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()