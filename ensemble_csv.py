import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from ensemble_boxes import weighted_boxes_fusion
from glob import glob


if __name__ == "__main__":
    # extract file_name from test image paths
    sources = sorted(glob("./datasets/dataset/test/*.png"))
    sources = [source.split("/")[-1] for source in sources]  # *.png

    # wbf ensemble
    iou_thr = 0.4
    skip_box_thr = 0.0001
    weights = None

    # image width, height
    img_w = 1920
    img_h = 1080

    results_dict = defaultdict(list)  # for submission

    data_frames = [
        pd.read_csv("./weights/10fold/fold0_epoch170.csv"),
        pd.read_csv("./weights/10fold/fold1_epoch160.csv"),
        pd.read_csv("./weights/10fold/fold2_epoch180.csv"),
        pd.read_csv("./weights/10fold/fold3_epoch170.csv"),
        pd.read_csv("./weights/10fold/fold4_epoch160.csv"),
        pd.read_csv("./weights/10fold/fold5_epoch170.csv"),
        pd.read_csv("./weights/10fold/fold6_epoch160.csv"),
        pd.read_csv("./weights/10fold/fold7_epoch180.csv"),
        pd.read_csv("./weights/10fold/fold8_epoch160.csv"),
        pd.read_csv("./weights/10fold/fold9_epoch190.csv"),
        
        pd.read_csv("./weights/10fold/fold0_last.csv"),
        pd.read_csv("./weights/10fold/fold1_last.csv"),
        pd.read_csv("./weights/10fold/fold2_last.csv"),
        pd.read_csv("./weights/10fold/fold3_last.csv"),
        pd.read_csv("./weights/10fold/fold4_last.csv"),
        pd.read_csv("./weights/10fold/fold5_last.csv"),
        pd.read_csv("./weights/10fold/fold6_last.csv"),
        pd.read_csv("./weights/10fold/fold7_last.csv"),
        pd.read_csv("./weights/10fold/fold8_last.csv"),
        pd.read_csv("./weights/10fold/fold9_last.csv"),
        
        pd.read_csv("./weights/10fold_aug/fold0_epoch180.csv"),
        pd.read_csv("./weights/10fold_aug/fold1_epoch170.csv"),
        pd.read_csv("./weights/10fold_aug/fold2_epoch180.csv"),
        pd.read_csv("./weights/10fold_aug/fold3_epoch180.csv"),
        pd.read_csv("./weights/10fold_aug/fold4_epoch160.csv"),
        pd.read_csv("./weights/10fold_aug/fold5_epoch160.csv"),
        pd.read_csv("./weights/10fold_aug/fold6_epoch170.csv"),
        pd.read_csv("./weights/10fold_aug/fold7_epoch180.csv"),
        pd.read_csv("./weights/10fold_aug/fold8_epoch170.csv"),
        pd.read_csv("./weights/10fold_aug/fold9_epoch170.csv"),
        
        pd.read_csv("./weights/10fold_aug/fold0_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold1_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold2_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold3_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold4_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold5_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold6_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold7_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold8_last.csv"),
        pd.read_csv("./weights/10fold_aug/fold9_last.csv"),
    ]

    for source in tqdm(sources):
        boxes_list, scores_list, labels_list = (
            [],
            [],
            [],
        )  # temporary result list for all dataframes per source
        for data_frame in data_frames:
            try:
                source_data_frame = data_frame.loc[
                    data_frame.file_name == source
                ].copy()
            except:  # no detection
                continue

            # bbox normalization (for wbf ensemble)
            source_data_frame[["point1_x", "point3_x"]] = source_data_frame[
                ["point1_x", "point3_x"]
            ].apply(lambda x: x / img_w)
            source_data_frame[["point1_y", "point3_y"]] = source_data_frame[
                ["point1_y", "point3_y"]
            ].apply(lambda y: y / img_h)

            bboxes = source_data_frame[
                ["point1_x", "point1_y", "point3_x", "point3_y"]
            ].values
            classes = source_data_frame.class_id.values
            scores = source_data_frame.confidence.values

            boxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(classes)

        wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        for idx in range(len(wbf_boxes)):
            results_dict["file_name"].append(source)
            results_dict["class_id"].append(int(wbf_labels[idx]))
            results_dict["confidence"].append(wbf_scores[idx])
            # bbox denormalization
            results_dict["point1_x"].append(int(wbf_boxes[idx][0] * img_w))
            results_dict["point1_y"].append(int(wbf_boxes[idx][1] * img_h))
            results_dict["point2_x"].append(int(wbf_boxes[idx][2] * img_w))
            results_dict["point2_y"].append(int(wbf_boxes[idx][1] * img_h))
            results_dict["point3_x"].append(int(wbf_boxes[idx][2] * img_w))
            results_dict["point3_y"].append(int(wbf_boxes[idx][3] * img_h))
            results_dict["point4_x"].append(int(wbf_boxes[idx][0] * img_w))
            results_dict["point4_y"].append(int(wbf_boxes[idx][3] * img_h))

    result_dataframe = pd.DataFrame(results_dict)
    result_dataframe.to_csv("submission.csv", index=False)
