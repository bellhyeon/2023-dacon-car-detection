# 합성데이터 기반 객체 탐지 AI 경진대회
합성 데이터를 활용한 자동차 탐지 AI 모델 개발
<br>[Competition Link](https://dacon.io/competitions/official/236107/overview/description)
* 주최: 비솔(VISOL)
* 주관: 데이콘
* **Private 6th, Score 0.98813**
***

## Structure
(normal) Train/Test data and sample submission file must be placed under **`datasets/dataset`** folder.  and Augmented train data must be placed under **`datasets/dataset_aug`** folder.
```
repo
  |——datasets
        |——dataset
                |——train
                        |——syn_00000.png
                        |——syn_00000.txt
                        |——....
                |——test
                        |——064442001.png
                        |——....
                |——classes.txt
                |——sample_submission.csv
        |——dataset_aug
                |——train
                        |——syn_00000.png
                        |——syn_00000.txt
                        |——....
  |——weights
        |——10fold
        |——10fold_aug                     
  |——yolov8
        |——dataset_cfg
                |——10fold
                    |——fold_0.yaml
                    |——....
                |——10fold_aug
                    |——fold_0.yaml
                    |——....
        |——training_cfg
                |——10fold
                    |——fold_0.yaml
                    |——....
                |——10fold_aug
                    |——fold_0.yaml
                    |——....
        |——fix_seed.py
        |——translation.py
        |——yolov8_10fold_train_1.py
        |——yolov8_10fold_train_2.py
        |——yolov8_10fold_aug_train_1.py
        |——yolov8_10fold_aug_train_2.py
        |——yolov8_single_inference.py
  augment_bboxes.py
  ensemble_csv.py
  fix_yaml_path.py
  inference_10fold.sh
  inference_ckpt.sh
  preprocess_yolo_data_kfold.py
  requirements.txt
```
***

## Development Environment
* Ubuntu 20.04
* AMD Ryzen Threadripper PRO 5995WX (64 cores, 128 threads)
* RTX 4090 2EA
* CUDA 11.8
* cuDNN 8.8.0
### [Option 1] Docker Image
```shell
docker pull kimjonghyeon/dacon-car
```
```shell
docker run -it --ipc=host -v {local_repo_path}:{container_path} --name container_name --gpus all kimjonghyeon/dacon-car
```
### [Option 2] Anaconda environment
[![Python 3.8.5](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-385/)

```shell
conda create -n env_name python=3.8.5
```
```shell
conda activate env_name
```
```shell
pip install --no-dependencies -r requirements.txt
```
***

## Download augmented datasets
Augmented datasets are available in here [dataset_aug.zip](https://bit.ly/dataset_aug).  
Downloading augmented datasets are also available on shell using wget.
```shell
wget https://bit.ly/dataset_aug -O dataset_aug.zip
mkdir datasets/
unzip dataset_aug.zip -d datasets/dataset_aug
```
## Download weights (for only inference)

Weights are available in here [weights.zip](https://bit.ly/3CIc2H2).  
Downloading weights are also available on shell using wget.
```shell
wget https://bit.ly/3CIc2H2 -O weights.zip
unzip weights.zip
```
***
## Solution

***
## How to Train
### 1. Prepare 10-fold dataset with pre-defined indicies using `preprocess_yolo_data_kfold.py`
**Arguments**
* `img_paths`: specify the original train image path (for glob)
* `txt_paths`: specify the original train text (bbox information) path (for glob)
* `save_abs_path`: path to save k-fold dataset
* `num_folds`: set number of folds
* `pickle_file_path`: pre-defined 10-fold indicies
* `from_scratch`: if "True", can make own fold

**For `normal` dataset**
```shell
python preprocess_yolo_data_kfold.py \
       --img_paths "./datasets/dataset/train/*.png" \
       --txt_paths "./datasets/dataset/train/*.txt" \
       --save_abs_path "./datasets/dataset/yolo" \
       --num_folds 10 \
       --pickle_file_path "./10fold_indicies.pkl" \
       --from_scratch "False"
```
**For `augmented` dataset**
```shell
python preprocess_yolo_data_kfold.py \
       --img_paths "./datasets/dataset_aug/train/*.png" \
       --txt_paths "./datasets/dataset_aug/train/*.txt" \
       --save_abs_path "./datasets/dataset_aug/yolo" \
       --num_folds 10 \
       --pickle_file_path "./10fold_indicies.pkl" \
       --from_scratch "False"
```
### 2. Fix absolute path to own absolute path
fix absolute path for train and dataset configuration. e.g., "/home/jonghyeon/dacon_car" (for conda env), "/dacon_car" (for docker env)
```shell
python fix_yaml_path.py --absolute_path "own_absolute_path"
```
### 3. Train
The model was trained on two GPUs, from fold 1 to fold 5 was trained on GPU no.0, and from fold 6 to fold 9 was trained on GPU no.1. <br>
If want to change GPU no., please change the `device` argument in the train config yaml files manually (yolov8/training_cfg/\*/\*.yaml).

**1) Train with normal `normal` dataset**
```shell
python yolov8/yolov8_10fold_train_1.py
```
```shell
python yolov8/yolov8_10fold_train_2.py
```
**2) Train with `augmented` dataset**
```shell
python yolov8/yolov8_10fold_aug_train_1.py
```
```shell
python yolov8/yolov8_10fold_aug_train_2.py
```

### 5. Inference
**[Option 1] Checkpoints inference**
```shell
sh inference_ckpt.sh
```
**[Option 2] Training inference**
```shell
sh inference_10fold.sh
```

### 6. Ensemble Inference Results
Can change fold results files to ensemble in `ensemble_csv.py`
```shell
python ensemble_csv.py
```