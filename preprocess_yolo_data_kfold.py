import os
from glob import glob
from typing import List
import numpy as np
from sklearn.model_selection import KFold
from shutil import copy2
import cv2
import argparse
import pickle
from tqdm import tqdm
from yolov8.translation import str2bool


def copy_image(src: str, dst: str):
    file_name = src.split("/")[-1]  # syn_xxxxx.png
    dst = os.path.join(dst, file_name)
    copy2(src, dst)


def labelme2yolo(txt_file_path: str, txt_save_folder: str):
    file_name = txt_file_path.split("/")[-1].split(".txt")[0]  # # syn_xxxxx

    img_path = txt_file_path.replace("txt", "png")
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape

    with open(txt_file_path, "r") as txt_file:
        lines_list = []
        lines = txt_file.readlines()
        for line in lines:
            line = list(map(float, line.strip().split(" ")))
            class_name = int(line[0])
            x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
            x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
            x_center, y_center = float(((x_min + x_max) / 2) / img_width), float(
                ((y_min + y_max) / 2) / img_height
            )
            width, height = (
                abs(x_max - x_min) / img_width,
                abs(y_max - y_min) / img_height,
            )
            lines_list.append([class_name, x_center, y_center, width, height])

    txt_save_path = os.path.join(txt_save_folder, f"{file_name}.txt")

    with open(txt_save_path, "w") as yolo_annot:
        for line in lines_list:
            yolo_annot.write(
                str(line[0])
                + " "
                + str(line[1])
                + " "
                + str(line[2])
                + " "
                + str(line[3])
                + " "
                + str(line[4])
                + "\n"
            )


def preprocess_yolo_data(
    save_abs_path: str, img_paths: np.ndarray, txt_paths: np.ndarray, num_folds: int
):
    print(f"Preparing {num_folds} fold dataset from scratch...")
    save_abs_path = os.path.join(save_abs_path, f"fold_{num_folds}")
    os.makedirs(save_abs_path, exist_ok=True)

    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    indicies: List = []

    for fold_num, (train_idx, val_idx) in enumerate(k_fold.split(img_paths, txt_paths)):
        print(f"Fold: {fold_num}")
        indicies.append([train_idx.tolist(), val_idx.tolist()])
        train_img_save_path = os.path.join(
            save_abs_path, str(fold_num), "images", "train"
        )
        train_txt_save_path = os.path.join(
            save_abs_path, str(fold_num), "labels", "train"
        )
        os.makedirs(train_img_save_path, exist_ok=True)
        os.makedirs(train_txt_save_path, exist_ok=True)

        valid_img_save_path = os.path.join(
            save_abs_path, str(fold_num), "images", "val"
        )
        valid_txt_save_path = os.path.join(
            save_abs_path, str(fold_num), "labels", "val"
        )
        os.makedirs(valid_img_save_path, exist_ok=True)
        os.makedirs(valid_txt_save_path, exist_ok=True)

        train_img_paths = img_paths[train_idx.astype(int)]
        train_txt_paths = txt_paths[train_idx.astype(int)]
        print(f"Start preparing fold {fold_num} train data...")

        for train_idx_ in tqdm(range(len(train_img_paths))):
            copy_image(train_img_paths[train_idx_], train_img_save_path)
            labelme2yolo(train_txt_paths[train_idx_], train_txt_save_path)

        print(f"Preparing fold {fold_num} training data done.")

        valid_img_paths = img_paths[val_idx.astype(int)]
        valid_txt_paths = txt_paths[val_idx.astype(int)]
        print(f"Start preparing fold {fold_num} validation data...")

        for valid_idx_ in tqdm(range(len(valid_img_paths))):
            copy_image(valid_img_paths[valid_idx_], valid_img_save_path)
            labelme2yolo(valid_txt_paths[valid_idx_], valid_txt_save_path)

        print(f"Preparing fold {fold_num} validation data done.")

    with open(f"{num_folds}fold_indicies.pkl", "wb") as f:
        pickle.dump(indicies, f)

    print(f"Saved {num_folds} fold indicies to pickle.")


def preprocess_yolo_data_with_pickle(
    save_abs_path: str,
    img_paths: np.ndarray,
    txt_paths: np.ndarray,
    pickle_file_path: str,
    num_folds: int,
):
    save_abs_path = os.path.join(save_abs_path, f"fold_{num_folds}")
    os.makedirs(save_abs_path, exist_ok=True)

    print(f"Preparing {num_folds} fold dataset from pre-defined indices...")

    with open(pickle_file_path, "rb") as f:
        indicies = pickle.load(f)

    for fold_num in range(num_folds):
        print(f"Fold: {fold_num}")
        train_img_save_path = os.path.join(
            save_abs_path, str(fold_num), "images", "train"
        )
        train_txt_save_path = os.path.join(
            save_abs_path, str(fold_num), "labels", "train"
        )
        os.makedirs(train_img_save_path, exist_ok=True)
        os.makedirs(train_txt_save_path, exist_ok=True)

        valid_img_save_path = os.path.join(
            save_abs_path, str(fold_num), "images", "val"
        )
        valid_txt_save_path = os.path.join(
            save_abs_path, str(fold_num), "labels", "val"
        )
        os.makedirs(valid_img_save_path, exist_ok=True)
        os.makedirs(valid_txt_save_path, exist_ok=True)

        train_img_paths = img_paths[indicies[fold_num][0]]
        train_txt_paths = txt_paths[indicies[fold_num][0]]

        print(f"Start preparing fold {fold_num} train data...")
        for train_idx_ in tqdm(range(len(train_img_paths))):
            copy_image(train_img_paths[train_idx_], train_img_save_path)
            labelme2yolo(train_txt_paths[train_idx_], train_txt_save_path)
        print(f"Preparing fold {fold_num} train data done.")

        print(f"Start preparing fold {fold_num} validation data...")
        valid_img_paths = img_paths[indicies[fold_num][1]]
        valid_txt_paths = txt_paths[indicies[fold_num][1]]
        for valid_idx_ in tqdm(range(len(valid_img_paths))):
            copy_image(valid_img_paths[valid_idx_], valid_img_save_path)
            labelme2yolo(valid_txt_paths[valid_idx_], valid_txt_save_path)
        print(f"Preparing fold {fold_num} validation data done.")

        print(f"Fold {fold_num} data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_paths",
        help="original image data absolute path. ./datasets/dataset/train/*.png or ./datasets/dataset_aug/train/*.png",
        type=str,
        required=True,
        default="./datasets/dataset/train/*.png",
    )
    parser.add_argument(
        "--txt_paths",
        help="original label data absolute path. ./datasets/dataset/train/*.txt or ./datasets/dataset_aug/train/*.txt",
        type=str,
        required=True,
        default="./datasets/dataset/train/*.txt",
    )
    parser.add_argument(
        "--save_abs_path",
        help="absolute folder path to save fold files. ./datasets/dataset/yolo or ./datasets/dataset_aug/yolo",
        type=str,
        required=True,
        default="./datasets/dataset/yolo",
    )
    parser.add_argument(
        "--num_folds", help="set number of k-folds", type=int, required=True
    )
    parser.add_argument(
        "--pickle_file_path",
        help="k-fold indicies pickle file path",
        type=str,
        default="./10fold_indicies.pkl",
    )
    parser.add_argument(
        "--from_scratch",
        help="preprocess data with custom indices",
        type=str2bool,
        default="False",
    )

    args = parser.parse_args()

    img_paths = np.array(
        sorted(glob(args.img_paths))
    )  # "./datasets/dataset/train/*.png" or "./datasets/dataset_aug/train/*.png"
    txt_paths = np.array(
        sorted(glob(args.txt_paths))
    )  # "./datasets/dataset/train/*.txt" or "./datasets/dataset_aug/train/*.txt"

    if args.from_scratch:
        preprocess_yolo_data(args.save_abs_path, img_paths, txt_paths, args.num_folds)
    else:
        preprocess_yolo_data_with_pickle(
            args.save_abs_path,
            img_paths,
            txt_paths,
            args.pickle_file_path,
            args.num_folds,
        )
