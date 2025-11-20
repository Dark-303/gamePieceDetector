from ultralytics import YOLO

'''
Current Models:
coral_modelVER0:
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
coral_modelVER1:
    - coral_modelVER1.1:
        Path: runs/train/coral_modelVER1/coral_modelVER1.1/weights/best.pt
        Epochs Trained: 300
        Changes: Retrained from scratch with 300 epochs, image size 612, added multiple detections
        Grade: Good performance, could use some more work with >1 corals, ok detection live
    - coral_modelVER1.2:
        Path: runs/train/coral_modelVER1/coral_modelVER1.2/weights/best.pt
        Epochs Trained: 150
        Changes: Retrained from scratch with 150 epochs, added nulls from art room
        Grade: Fairly accurate, detects most corals, struggles with chair legs
    - coral_modelVER1.3:
        Path: runs/train/coral_modelVER1/coral_modelVER1.3/weights/best.pt
        Epochs: Trained: 450
        Changes: Evolved VER1.1, added nulls from art room
        Grade: Very good performance, detects multiple corals well, minor issues with false positives
'''
model = YOLO("runs/train/coral_modelVER1/coral_modelVER1.2/weights/best.pt")

results = model(
    "dataset/images/val", # Image folder path
    save=True,
    project="tests/coral_modelVER1/coral_modelVER1.2_detect",  # Folder to save results
    name="predict01", # Subfolder name
    exist_ok=True # Overwrite if exists...
)

print("Test completed. See file explore for results.")