from ultralytics import YOLO
from fix_seed import seed_everything

if __name__ == "__main__":
    seed_everything(seed=42)

    # Load the model.
    # Training.
    for i in range(5):
        model = YOLO("yolov8l.pt")
        model.train(cfg=f"yolov8/training_cfg/10fold/fold_{i}.yaml")
