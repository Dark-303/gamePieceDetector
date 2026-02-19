from ultralytics import YOLO

# Load YOLOv5nu model
model = YOLO("yolov5nu.pt")

# Export to ONNX
model.export(format="onnx")  # Creates .onnx file in the current directory
print("ONNX export complete!")
