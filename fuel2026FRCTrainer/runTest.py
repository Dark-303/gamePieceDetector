from ultralytics import YOLO

'''
Current Models:
None
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