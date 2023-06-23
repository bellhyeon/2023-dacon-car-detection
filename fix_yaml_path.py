import argparse
import yaml
from glob import glob
import os


def fix_path(absolute_path, training_cfg_paths, dataset_cfg_paths):
    for training_cfg_path in training_cfg_paths:
        with open(training_cfg_path) as training_yaml_file:
            training_cfg = yaml.safe_load(training_yaml_file)
            origin_dataset_yaml_path = training_cfg["data"]
            dataset_yaml_path = "yolov8" + training_cfg["data"].split("yolov8")[-1]

            abs_dataset_yaml_path = os.path.join(absolute_path, dataset_yaml_path)

            training_cfg["data"] = abs_dataset_yaml_path

            with open(training_cfg_path, "w") as modified_training_yaml:
                yaml.safe_dump(training_cfg, modified_training_yaml, sort_keys=False)

            print(
                f"Change dataset yaml path {origin_dataset_yaml_path} to {abs_dataset_yaml_path}"
            )

    for dataset_cfg_path in dataset_cfg_paths:
        with open(dataset_cfg_path) as dataset_yaml_file:
            dataset_cfg = yaml.safe_load(dataset_yaml_file)
            dataset_path = "datasets" + dataset_cfg["path"].split("datasets")[-1]

            origin_path = dataset_cfg["path"]

            abs_dataset_path = os.path.join(absolute_path, dataset_path)

            dataset_cfg["path"] = abs_dataset_path

            with open(dataset_cfg_path, "w") as modified_dataset_yaml:
                yaml.safe_dump(dataset_cfg, modified_dataset_yaml, sort_keys=False)

            print(f"Change dataset path {origin_path} to {abs_dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--absolute_path",
        type=str,
        required=True,
        help="absolute repo path on local or container. e.g., /home/jonghyeon/dacon-car (local) or /dacon-car (container)",
    )
    args = parser.parse_args()

    training_cfg_paths = sorted(glob("./yolov8/training_cfg/*/*.yaml"))
    dataset_cfg_paths = sorted(glob("./yolov8/dataset_cfg/*/*.yaml"))

    fix_path(args.absolute_path, training_cfg_paths, dataset_cfg_paths)
