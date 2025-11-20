import threading
from trainer import Trainer

# Coral Detector has been discontinued. Use multiprocessing instead to load graphs.

# Create instances of Trainer for each model
trainer1 = Trainer("yolov5nu.pt")
trainer2 = Trainer("runs/train/coral_modelVER1/coral_modelVER1.1/weights/best.pt")

# Set up threads
thread1 = threading.Thread(target=trainer1.train_model, args=(1, 2, 150, 612)) # Train from scratch
thread2 = threading.Thread(target=trainer2.train_model, args=(1, 3, 150, 612)) # Evolve from VER1.1

# Start threads
thread1.start()
thread2.start()

# Wait for both to finish
thread1.join()
thread2.join()

print("Both trainings complete!")