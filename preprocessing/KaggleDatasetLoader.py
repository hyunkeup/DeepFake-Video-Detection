from typing import Any

import cv2
import numpy as np
from numpy import ndarray, dtype

from preprocessing.DatasetLoader import DatasetLoader


class KaggleDatasetLoader(DatasetLoader):
    def __init__(self):
        super().__init__()
        self.frame_shape = (224, 224, 3)
        self.num_of_frames = 300

    def load(self, directory_path: str):
        # Read the metadata.json
        metadata_path = f"{directory_path}/metadata.json"
        metadata = self.read_metadata(metadata_path)

        # Load video
        train_x = []
        for (filename, _) in metadata:
            video_path = f"{directory_path}/{filename}"
            video_frames = self.read_video(video_path)
            train_x.append(video_frames)

        train_x = np.array(train_x)
        train_y = metadata[:, 1]

        # Save loaded data
        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)

        return train_x, train_y

    def read_metadata(self, metadata_path: str) -> ndarray[Any, dtype[Any]]:
        metadata = []
        data = self.read_json_file(metadata_path)
        for filename in data:
            metadata.append((filename, "0" if data[filename]["label"] == "FAKE" else "1"))

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

            video_frames.append(frame)

        num_of_frames = len(video_frames)
        if num_of_frames < self.num_of_frames:
            for _ in range(self.num_of_frames - num_of_frames):
                dummy_frame = np.zeros(self.frame_shape, dtype=np.uint8)
                video_frames.append(dummy_frame)

        return np.array(video_frames)
