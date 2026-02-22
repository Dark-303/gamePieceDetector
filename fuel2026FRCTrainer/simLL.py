import cv2
from ultralytics import YOLO
import ntcore
import json
import time
import sys

# --- CONFIGURATION ---
NT_SERVER = "127.0.0.1"
TABLE_NAME = "limelight-intake"
MODEL_PATH = "C:/Users/ICung/FRC/CodeProjects/theGreatDetectorOfGameFRCPieces/fuel2026FRCTrainer/versions/fuel_modelVER1.0/fuel_modelVER1.0.pt"
H_FOV, V_FOV = 60, 45 

# 1. Setup NetworkTables
inst = ntcore.NetworkTableInstance.getDefault()
inst.setServer(NT_SERVER) 
inst.startClient4("LimelightSim")
table = inst.getTable(TABLE_NAME)

# Publishers
json_pub = table.getStringTopic("json").publish()
tv_pub = table.getBooleanTopic("tv").publish()
tx_pub = table.getDoubleTopic("tx").publish()
ta_pub = table.getDoubleTopic("ta").publish()
pipe_pub = table.getDoubleTopic("pipeline").publish()

print("Waiting for NT connection...", flush=True)
for _ in range(50):
    if inst.isConnected(): break
    time.sleep(0.1)

# 2. Load model
print("Loading model...", flush=True)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera failed.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret: continue
    
    # Run inference
    results = model(frame, verbose=False)[0]
    detected_objects = []
    h, w, _ = frame.shape
    center_x, center_y = w / 2, h / 2

    for box in results.boxes:
        # xywh: 0=x_center, 1=y_center, 2=width, 3=height
        b = box.xywh[0].cpu().numpy()
        conf = float(box.conf.item())
        cls_id = float(box.cls.item())
        
        # Calculate offsets in degrees
        calc_tx = (b[0] - center_x) * (H_FOV / w)
        calc_ty = (b[1] - center_y) * (V_FOV / h)
        calc_ta = (b[2] * b[3]) / (w * h) * 100
        
        # Structure for LimelightHelpers DetectorResult
        obj_data = {
            "class": "balls",          # Maps to className
            "classID": float(cls_id),  # Maps to classID
            "conf": float(conf),       # Maps to confidence
            "ta": float(calc_ta),      # Maps to ta
            "tx": float(calc_tx),      # Maps to tx
            "ty": float(calc_ty),      # Maps to ty
            "txp": 0.0,                # Maps to tx_pixels (REQUIRED)
            "typ": 0.0,                # Maps to ty_pixels (REQUIRED)
            "tx_nocross": float(calc_tx), # Maps to tx_nocrosshair (REQUIRED)
            "ty_nocross": float(calc_ty)  # Maps to ty_nocrosshair (REQUIRED)
        }
        detected_objects.append(obj_data)

    # Wrap in the "Results" object LimelightHelpers expects
    ll_json = {
        "Results": {
            "pID": 1.0, # Pipeline ID
            "v": 1 if detected_objects else 0,
            "ts": float(time.time() * 1000), 
            "Detector": detected_objects,
            "Retro": [],
            "Fiducial": [],
            "Classifier": [],
            "Barcode": []
        }
    }

    # 3. Publish
    json_pub.set(json.dumps(ll_json))
    pipe_pub.set(1.0)
    
    has_target = len(detected_objects) > 0
    tv_pub.set(has_target)
    if has_target:
        tx_pub.set(detected_objects[0]["tx"])
        ta_pub.set(detected_objects[0]["ta"])

    cv2.imshow("Limelight Sim", results.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()