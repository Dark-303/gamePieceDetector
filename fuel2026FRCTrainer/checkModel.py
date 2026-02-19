from ultralytics import YOLO

# May take a few moments to load
model = YOLO("yolov5nu.pt")
print("Loading Model...")
print(model.model)  # prints model info
print("Model Loaded Successfully.")