import os
import time
from typing import Any

import cv2
import numpy as np
from numpy import ndarray, dtype

from preprocessing.DatasetsLoader import DatasetsLoader


class DfdcDatasetsLoader(DatasetsLoader):
    def __init__(self, frame_shape: tuple = (224, 224, 3), num_of_frames: int = 300):
        """
        :param frame_shape: (224, 224, 1) or (224, 224, 3), you can also use 299x299, 331x331, 448x448
        :param num_of_frames: the number of frames in a video.
                * If there are less than num_of_frames, then add more dummy frames.
                * If there are grater than num_of_frames, then get only num_of_frames.
        """
        super().__init__()
        self.frame_shape = frame_shape
        self.num_of_frames = num_of_frames

    def load(self, directory_path: str):
        """
        :param directory_path: C:\workspace\deepfake-detection-challenge
        :return: train_y, train_x
        """
        print("=" * 50 + " Loading video datasets from DFDC " + "=" * 50)

        directory_name = os.path.basename(directory_path)
        frame_info = '_'.join([str(x) for x in list(self.frame_shape)])
        np_train_y_path = f"{directory_path}/{directory_name}_{frame_info}_train_y.npy"
        np_train_x_path = f"{directory_path}/{directory_name}_{frame_info}_train_x.npy"
        if os.path.exists(np_train_x_path) and os.path.exists(np_train_y_path):
            s_time = time.time()
            train_y = np.load(np_train_y_path)
            train_x = np.load(np_train_x_path)
            e_time = time.time()
            print(f"* Loading time for .npy: {e_time - s_time} seconds.")
            print(f"\t* train_y: {train_y.shape}")
            print(f"\t* train_x: {train_x.shape}")
            return train_y, train_x

        # Read the metadata.json
        s_time = time.time()
        metadata_path = f"{directory_path}/metadata.json"
        metadata = self.read_metadata(metadata_path)
        train_y = metadata[:, 1]
        e_time = time.time()
        print(f"* Loading time for metadata: {e_time - s_time} seconds.")

        # Load video
        s_time = time.time()
        train_x = []
        for (filename, _) in metadata:
            video_path = f"{directory_path}/{filename}"
            video_frames = self.read_video(video_path)
            train_x.append(video_frames)

        train_x = np.array(train_x)
        e_time = time.time()
        print(f"* Loading time for videos: {e_time - s_time} seconds.")

        # Save loaded data
        s_time = time.time()
        np.save(np_train_y_path, train_y)
        np.save(np_train_x_path, train_x)
        e_time = time.time()
        print(f"* Saving time for videos: {e_time - s_time} seconds.")
        print(f"\t* train_y: {train_y.shape}")
        print(f"\t* train_x: {train_x.shape}")

        return train_y, train_x

    def read_metadata(self, metadata_path: str) -> ndarray[Any, dtype[Any]]:
        """
        :param metadata_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\metadata.json
        :return: [["filename.mp4", "0 or 1"]]
        """
        metadata = []
        data = self.read_json_file(metadata_path)
        for filename in data:
            metadata.append((filename, "0" if data[filename]["label"] == "REAL" else "1"))

        return np.array(metadata)

    def read_video(self, video_path: str) -> ndarray[Any, dtype[Any]]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_frames = []
        for frame_number in range(total_frames):
            if len(video_frames) == self.num_of_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_shape is not None:
                frame = cv2.resize(frame, (self.frame_shape[0], self.frame_shape[1]))
                if self.frame_shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            video_frames.append(frame)

        num_of_frames = len(video_frames)
        if num_of_frames < self.num_of_frames:
            for _ in range(self.num_of_frames - num_of_frames):
                dummy_frame = np.zeros(self.frame_shape, dtype=np.uint8)
                video_frames.append(dummy_frame)

        return np.array(video_frames)
