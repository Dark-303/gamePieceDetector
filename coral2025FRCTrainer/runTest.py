from ultralytics import YOLO

'''
Current Models:
- coral_modelVER1: 
    Path: runs/train/coral_modelVER1/weights/best.pt
    Epochs Trained: 50
    Grade: Satisfactory for initial testing
- coral_modelVER2:
    Path: runs/train/coral_modelVER2/weights/best.pt
    Epochs Trained: 100
    Grade: TBD - Training
'''
model = YOLO("runs/train/coral_modelVER1/weights/best.pt")

results = model(
    "dataset/images/val/",
    save=True,
    project="coral_modelVER1_detect",  # your custom folder
    name="predict02",                    # subfolder name
    exist_ok=True                      # prevents YOLO from adding numbers
)

print("Detection attempt completed. See file explore for results.")