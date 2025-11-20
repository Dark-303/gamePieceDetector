# 🪸 Coral Detection 🪸

YOLOv5 models trained on reefscape coral images. If you need .onnx files use [converter.py](coral2025FRCTrainer/converter.py).

This project uses **openCV** and **Ultralytics**.

**coral2025FRCTrainer is no longer being developed and there is no java implementation**

## Dowloads
| Version | Download Link | Training Info | Notes |
|--------|----------------|----------------|-------|
| **VER 1.3** | [coral_modelVER1.3](coral2025FRCTrainer/versions/coral_modelVER1.3/) | 450 epochs | Best on carpet |
| **VER 0.3** | [coral_modelVER0.3](coral2025FRCTrainer/versions/coral_modelVER0.3/) | 100 epochs | Detects coral only horizontally |
| **VER 1.2** | [coral_modelVER1.2](coral2025FRCTrainer/versions/coral_modelVER1.2/) | 150 epochs | Detects most positions but may pick up on other objects |

## Best Releases from VER0 and VER1
- [Version 1.3](coral2025FRCTrainer/versions/coral_modelVER1.3/) 
    - Trained for 450 epochs
    - Works best on carpet
- [Version 0.3](coral2025FRCTrainer/versions/coral_modelVER0.3/) 
    - Trained for 100 epochs
    - Detects coral only horizontally

## Broader more general detection for VER1
- [Version 1.2](coral2025FRCTrainer/versions/coral_modelVER1.2/) 
    - trained for 150 epochs
    - Works for most coral positions but picks up on other objects
