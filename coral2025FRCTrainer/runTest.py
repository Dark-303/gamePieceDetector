from ultralytics import YOLO

'''
Current Models:
- coral_modelVER0.1: 
    Path: runs/train/coral_modelVER0/coral_modelVER1/weights/best.pt
    Epochs Trained: 50
    Changes: N/A
    Grade: Satisfactory for initial testing
- coral_modelVER0.2:
    Path: runs/train/coral_modelVER0/coral_modelVER2/weights/best.pt
    Epochs Trained: 100
    Changes: Increased epochs
    Grade: Good performance but can be improved in most areas
- coral_modelVER0.3:
    Path: versions/coral_modelVER0.3.pt
    Epochs Trained: 200
    Changes: Evolved VER2 with more epochs, increased image size to 612
    Grade: Fairly accurate, cannot detect verticals and certain angles
'''
model = YOLO("runs/train/coral_modelVER1/coral_modelVER1.1/weights/best.pt")

results = model(
    "dataset/images/val",
    save=True,
    project="tests/coral_modelVER1/coral_modelVER1.1_detect",  # your custom folder
    name="predict01",                    # subfolder name
    exist_ok=True                      # prevents YOLO from adding numbers
)

print("Detection attempt completed. See file explore for results.")