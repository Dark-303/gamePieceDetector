from trainer import Trainer

# 1. We need a helper function that can be "pickled" (serialized) 
# to be sent to a different CPU process.
def run_training(model_path, version, subversion, epochs, img_size):
    trainer = Trainer(model_path)
    trainer.train_model(version, subversion, epochs, img_size)

if __name__ == "__main__":
    run_training("yolov5nu.pt", 1, 0, 150, 640)