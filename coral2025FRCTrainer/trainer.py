from ultralytics import YOLO

# Load model (pretrained YOLOv5n)
model = YOLO("yolov5nu.pt")  # or yolov5nu.onnx if using ONNX as starting point

# Train the model
model.train(
    data="dataset/data.yaml",  # your data.yaml path
    imgsz=416,                 # input image size
    epochs=50,                 # adjust for your needs
    batch=4,                  # adjust depending on your GPU/CPU
    project="runs/train",      # folder to save results
    name="coral_modelVER1",   # subfolder name
    exist_ok=True
)