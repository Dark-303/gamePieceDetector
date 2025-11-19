from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/train/coral_modelVER1/weights/best.pt")

# Run live inference from your webcam
# 0 usually refers to the default webcam
results = model(source=0, show=True)  # show=True opens a window with live detections