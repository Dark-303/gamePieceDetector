from ultralytics import YOLO

class Trainer:
    def __init__(self, model_path="yolov5nu.pt"):
        self.model = YOLO(model_path)
        # Load default YOLOv5nu model

    def train_model(self, version, subversion, tarEpochs = 100, imgSize = 612):
        # Train the model
        self.model.train(
            data="dataset/data.yaml",  # your data.yaml path
            imgsz=612,                 # input image size
            epochs=tarEpochs,                 # adjust for your needs
            batch=4,                  # adjust depending on your GPU/CPU
            project=f"runs/train/coral_modelVER{version}",      # folder to save results
            name=f"coral_modelVER{version}.{subversion}",   # subfolder name
            exist_ok=True
        )