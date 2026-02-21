import multiprocessing
from trainer import Trainer

# 1. We need a helper function that can be "pickled" (serialized) 
# to be sent to a different CPU process.
def run_training(model_path, version, subversion, epochs, img_size):
    trainer = Trainer(model_path)
    trainer.train_model(version, subversion, epochs, img_size)

if __name__ == "__main__":
    # 2. Define your training configurations
    # Trainer 1: From scratch
    args1 = ("yolov5nu.pt", 1, 2, 150, 612)
    
    # Trainer 2: Evolving from VER 1.1
    args2 = ("runs/train/coral_modelVER1/coral_modelVER1.1/weights/best.pt", 1, 3, 150, 612)

    # 3. Create the Processes
    process1 = multiprocessing.Process(target=run_training, args=args1)
    process2 = multiprocessing.Process(target=run_training, args=args2)

    print("Starting parallel training sessions...")
    
    # 4. Start them at the same time
    process1.start()
    process2.start()

    # 5. Wait for both to finish
    process1.join()
    process2.join()

    print("Both trainings are complete and saved separately!")