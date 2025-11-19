from ultralytics import YOLO

# Load default YOLOv5n model
#model = YOLO("yolov5nu.pt")  # or yolov5nu.onnx if using ONNX as starting point

# Evolve model from an existing trained model
model = YOLO("runs/train/coral_modelVER2/weights/best.pt")

# Train the model
model.train(
    data="dataset/data.yaml",  # your data.yaml path
    imgsz=612,                 # input image size
    epochs=100,                 # adjust for your needs
    batch=4,                  # adjust depending on your GPU/CPU
    project="runs/train",      # folder to save results
    name="coral_modelVER3",   # subfolder name
    exist_ok=True
)