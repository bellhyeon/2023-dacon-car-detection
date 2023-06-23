from ultralytics import YOLO
from glob import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import argparse
from translation import str2bool
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_saved_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--fold_num", type=int, required=True)
    parser.add_argument("--tta", type=str2bool, default="False")
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=1920)
    args = parser.parse_args()

    print(
        f"Fold: {args.fold_num} | Model: {'Augmented' if 'aug' in args.model_saved_path else 'Normal'} / {args.model_saved_path.split('/')[-1]}"
    )
    # Load the YOLOv8 model
    model = YOLO(args.model_saved_path)
    sources = sorted(glob(args.test_data_path))

    results_dict = defaultdict(list)
    no_detections = []
    for source in tqdm(sources):
        file_name = source.split("/")[-1]

        results = model.predict(
            source,
            imgsz=args.img_size,
            device=args.device,
            augment=args.tta,
            verbose=False,
        )

        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        conf = results[0].boxes.conf.detach().cpu().numpy()
        cls = results[0].boxes.cls.detach().cpu().numpy()

        if len(boxes) == 0:  # no detection
            continue
        else:
            for idx in range(len(boxes)):
                results_dict["file_name"].append(file_name)
                results_dict["class_id"].append(cls[idx])
                results_dict["confidence"].append(conf[idx])
                results_dict["point1_x"].append(boxes[idx][0])
                results_dict["point1_y"].append(boxes[idx][1])
                results_dict["point2_x"].append(boxes[idx][2])
                results_dict["point2_y"].append(boxes[idx][1])
                results_dict["point3_x"].append(boxes[idx][2])
                results_dict["point3_y"].append(boxes[idx][3])
                results_dict["point4_x"].append(boxes[idx][0])
                results_dict["point4_y"].append(boxes[idx][3])

    csv_save_folder = "/".join(args.model_saved_path.split("/")[:3])
    epoch = args.model_saved_path.split("/")[-1].split(".pt")[0]
    if args.tta:
        csv_save_name = f"fold{args.fold_num}_{epoch}_tta.csv"
    else:
        csv_save_name = f"fold{args.fold_num}_{epoch}.csv"

    csv_save_path = os.path.join(csv_save_folder, csv_save_name)
    result_dataframe = pd.DataFrame(results_dict)
    result_dataframe.to_csv(csv_save_path, index=False)
