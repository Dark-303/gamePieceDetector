from ultralytics import YOLO

class Trainer:
    def __init__(self, model_path="yolov5nu.pt"):
        self.model = YOLO(model_path)
        # Load default YOLOv5nu model

    def train_model(self, version, subversion, tarEpochs = 100, imgSize = 612):
        # Train the model
        self.model.train(
            data="dataset/data.yaml", # data.yaml path for dataset
            imgsz=imgSize, # Image size
            epochs=tarEpochs, # Number of epochs
            batch=4, # Adjust based on computer capabilities            
            # Version naming control | coral_modelVER1/coral_modelVER1.1
            project=f"runs/train/coral_modelVER{version}", # Folder to save results using version
            name=f"coral_modelVER{version}.{subversion}", # subfolder name based on subversion
            exist_ok=True # Overwrite if exists...
        )