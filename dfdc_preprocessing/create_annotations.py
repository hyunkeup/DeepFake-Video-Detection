# -*- coding: utf-8 -*-
import glob
import os

import numpy as np

from dfdc_preprocessing.dfdc_args import get_args


def add_annotation(datasets, dataset_type):
    for filename, label in datasets:
        with open(ANNOTATION_PATH, 'a') as f:
            prefix_path = f"{os.path.join(ROOT_DIR, label, filename)}"
            video_path = f"{prefix_path}_face_cropped.npy"
            audio_path = f"{prefix_path}_audio_cropped.npy"
            row = f"{video_path};{audio_path};{dataset_type}\n"
            f.write(row)


def main():
    if os.path.exists(ANNOTATION_PATH):
        os.remove(ANNOTATION_PATH)

    train_datasets = []
    test_datasets = []
    validation_datasets = []
    for label in os.listdir(ROOT_DIR):  # real, fake
        directory_path = os.path.join(ROOT_DIR, label)
        target_datasets = []
        for file in glob.glob(os.path.join(directory_path, "*_face_cropped.npy")):
            video_filename = os.path.basename(file).split("_")[0]
            target_datasets.append((video_filename, label))

        # train: 60%, test: 20%, validation: 20%
        train, validate, test = np.split(np.array(target_datasets),
                                         [int(len(target_datasets) * 0.6), int(len(target_datasets) * 0.8)])

        train_datasets.extend(train)
        test_datasets.extend(test)
        validation_datasets.extend(validation_datasets)

    add_annotation(train_datasets, "training")
    add_annotation(test_datasets, "testing")
    add_annotation(validation_datasets, "validation")


if __name__ == "__main__":
    args = get_args()

    ROOT_DIR = args.root_dir
    ANNOTATION_PATH = os.path.join(ROOT_DIR, "annotations.txt")

    main()
