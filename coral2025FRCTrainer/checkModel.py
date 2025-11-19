from ultralytics import YOLO
model = YOLO("yolov5nu.pt")
print("Loading Model...")
print(model.model)  # prints model info; look for pretrained weights reference
print("Model Loaded Successfully.")