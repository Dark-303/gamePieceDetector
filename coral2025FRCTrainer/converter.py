from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO("yolov5n.pt")  # or yolov5n-tiny.pt if you want faster inference

# Export to ONNX
model.export(format="onnx")  # creates yolov5n.onnx
print("ONNX export complete!")
