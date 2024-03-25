# -*- coding: utf-8 -*-
import os
import json

def read_json_file(file_path):
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


root = '../datasets/raw_dataset/split_dataset'
root = os.path.abspath(root)
metadata_path = '../datasets/raw_dataset/final_metadata.json'
annotation_file = '../annotations.txt'
metadata_json_data = read_json_file(metadata_path)


for i, dataset_type in enumerate(os.listdir(root)):
    print(dataset_type)

    for video in os.listdir(os.path.join(root, dataset_type)):
        try:
            if not video.endswith('.npy') or 'croppad' not in video:
                continue
            video_key_name = video.split('_')[0] + '.mp4'
            label = metadata_json_data[video_key_name]['label']
            if label == "REAL":
                label = 1
            else:
                label = 0
            audio = video.split('_face')[0] + '_croppad.wav'
            if dataset_type == "train":
                with open(annotation_file, 'a') as f:
                    f.write(os.path.join(root, dataset_type, video) + ';' + os.path.join(root, dataset_type, audio) + ';' + str(label) + ';training' + '\n')

            elif dataset_type == "val":
                with open(annotation_file, 'a') as f:
                    f.write(os.path.join(root, dataset_type, video) + ';' + os.path.join(root, dataset_type, audio) + ';' + str(label) + ';validation' + '\n')

            else:
                with open(annotation_file, 'a') as f:
                    f.write(os.path.join(root, dataset_type, video) + ';' + os.path.join(root, dataset_type, audio) + ';'+ str(label) + ';testing' + '\n')

        except Exception as e:
            print("ERROR:", e, "file: ", video)