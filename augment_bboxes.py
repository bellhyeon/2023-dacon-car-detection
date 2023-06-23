import cv2
from albumentations import *
from glob import glob
import shutil
from tqdm import tqdm
import os


def get_partial_bboxes(image_path, annotation_path):
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes_coords = []
    partial_images = []

    # get bboxes from text file
    with open(annotation_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        bboxes = list(map(float, line.strip().split(" ")))
        x_min, y_min = int(round(bboxes[1])), int(round(bboxes[2]))
        x_max, y_max = int(round(max(bboxes[3], bboxes[5], bboxes[7]))), int(
            round(max(bboxes[4], bboxes[6], bboxes[8]))
        )

        bboxes_coords.append([x_min, x_max, y_min, y_max])
        partial_images.append(image[y_min:y_max, x_min:x_max])

    return bboxes_coords, partial_images


def albumentations_bbox_augmentation(image_path, bboxes, partial_imgs):
    file_name = image_path.split("/")[-1]  # syn_xxxxx.png

    org_image = cv2.imread(image_path)
    org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

    transform = Compose(
        [
            ToGray(p=0.01),
            GaussNoise(p=0.5),
            OneOf(
                [
                    MotionBlur(p=0.5),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ],
                p=0.5,
            ),
            OneOf([CLAHE(), Sharpen(), Emboss()], p=0.5),
            OneOf(
                [
                    HueSaturationValue(p=0.2),
                    RandomBrightnessContrast(p=0.2),
                ],
                p=0.2,
            ),
        ]
    )

    for idx in range(len(partial_imgs)):
        transformed = transform(image=partial_imgs[idx])
        transformed_image = transformed["image"]
        partial_imgs[idx] = transformed_image

    for idx in range(len(bboxes)):
        x_min, x_max = bboxes[idx][0], bboxes[idx][1]
        y_min, y_max = bboxes[idx][2], bboxes[idx][3]

        org_image[y_min:y_max, x_min:x_max] = partial_imgs[idx]

    org_image = cv2.cvtColor(org_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"./datasets/dataset_aug/train/{file_name}", org_image)
    shutil.copy2(
        f"./datasets/dataset/train/{file_name.split('.png')[0]}.txt",
        f"./datasets/dataset_aug/train/{file_name.split('.png')[0]}.txt",
    )


if __name__ == "__main__":
    os.makedirs("./datasets/dataset_aug/train", exist_ok=True)

    images = sorted(glob("./datasets/dataset/train/*.png"))
    labels = sorted(glob("./datasets/dataset/train/*.txt"))

    for idx in tqdm(range(len(images))):
        bboxes, partial_images = get_partial_bboxes(images[idx], labels[idx])
        albumentations_bbox_augmentation(images[idx], bboxes, partial_images)
