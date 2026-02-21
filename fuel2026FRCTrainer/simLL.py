import cv2
from ultralytics import YOLO
import ntcore # You may need to: pip install pyntcore

# 1. Setup NetworkTables
inst = ntcore.NetworkTableInstance.getDefault()
table = inst.getTable("limelight") # Your Java code looks here
inst.startClientDevicehub("127.0.0.1") # Point to your own computer (Sim)

# 2. Load your model
model = YOLO("versions/coral_modelVER1.3/coral_modelVER1.3.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)[0]

    if len(results.boxes) > 0:
        # Get the best target (closest to center)
        box = results.boxes[0] 
        # Convert pixel coordinates to "degrees" (rough estimate)
        # Assuming 640x480 webcam
        tx = (box.xywh[0][0].item() - 320) * (60/640) 
        ta = (box.xywh[0][2].item() * box.xywh[0][3].item()) / (640*480) * 100

        # Send to "Limelight" table
        table.putNumber("tv", 1)
        table.putNumber("tx", tx)
        table.putNumber("ta", ta)
    else:
        table.putNumber("tv", 0)

    cv2.imshow("Limelight Sim", results.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"): break