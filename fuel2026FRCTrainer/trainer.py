from ultralytics import YOLO

class Trainer:
    def __init__(self, model_path="yolov5nu.pt"):
        self.model = YOLO(model_path)
        # Load default YOLOv5nu model

    def train_model(self, version, subversion, tarEpochs=100, imgsz=640):
            # All parameters below match the 'overrides' dict in the source
            self.model.train(
                data="dataset/data.yaml", 
                imgsz=imgsz,                # Matched to 640 for stride 32
                epochs=tarEpochs, 
                batch=-1,                   # Enabled AutoBatch for efficiency
                
                # Enable augmentation
                augment = True,
                
                # Save Management
                project=f"runs/train/fuel_modelVER{version}", 
                name=f"fuel_modelVER{version}.{subversion}", 
                exist_ok=True,
                
                # Optimization for Limelight 4 CPU
                cache=True,                 # Loads images to RAM for faster training
                deterministic=True          # Ensures repeatable results
            )