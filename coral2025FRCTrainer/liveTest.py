import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/train/coral_modelVER1/coral_modelVER1.1/weights/best.pt")

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# --- Increase camera resolution here ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # or 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # or 720
# --------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

    # Visualize result on the frame
    annotated_frame = results[0].plot()

    # Show window
    cv2.imshow("Coral Detector Live", annotated_frame)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
