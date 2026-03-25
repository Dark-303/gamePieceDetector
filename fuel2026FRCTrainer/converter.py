from ultralytics import YOLO

# Load YOLOv5nu model
model = YOLO("versions/fuel_modelVER1.0/fuel_modelVER1.0.pt")

# Export to ONNX
model.export(format='tflite', imgsz=320, int8=True, nms=True)
print("tflite export complete!")
