from ultralytics import YOLO

'''
Current Models:
- coral_modelVER1: 
    Path: runs/train/coral_modelVER1/weights/best.pt
    Epochs Trained: 50
    Changes: N/A
    Grade: Satisfactory for initial testing
- coral_modelVER2:
    Path: runs/train/coral_modelVER2/weights/best.pt
    Epochs Trained: 100
    Changes: Increased epochs
    Grade: Good performance but can be improved in most areas
- coral_modelVER3:
    Path: runs/train/coral_modelVER3/weights/best.pt
    Epochs Trained: 200
    Changes: Evolved VER2 with more epochs, increased image size to 612
    Grade: TBD - Training
'''
model = YOLO("runs/train/coral_modelVER3/weights/best.pt")

results = model(
    "dataset/images/train",
    save=True,
    project="coral_modelVER3_detect",  # your custom folder
    name="predict02",                    # subfolder name
    exist_ok=True                      # prevents YOLO from adding numbers
)

print("Detection attempt completed. See file explore for results.")