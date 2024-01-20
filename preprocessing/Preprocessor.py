import os

import cv2
import dotenv
import numpy as np

import FileUtils

dotenv.load_dotenv()

FRAME_SHAPE: tuple = tuple([int(x) for x in os.environ.get("FRAME_SHAPE").split(",")])
NUM_OF_FRAMES: int = int(os.environ.get("NUM_OF_FRAMES"))


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def read_metadata(metadata_path: str) -> list:
        """
        :param metadata_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\metadata.json
        :return: [["filename.mp4", "REAL" or "FAKE"]]
        """
        metadata = []
        data = FileUtils.read_json_file(metadata_path)
        for filename in data:
            metadata.append((filename, data[filename]["label"]))

        return metadata

    @staticmethod
    def read_video(video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        num_of_video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_of_required_frames = NUM_OF_FRAMES ** 2
        if num_of_video_total_frames < num_of_required_frames:
            raise Exception(
                f"The num_of_video_total_frames must be greater than num_of_required_frames. num_of_video_total_frames: {num_of_video_total_frames}, num_of_required_frames: {NUM_OF_FRAMES}")

        video_frames = []
        interval = num_of_video_total_frames // num_of_required_frames
        target_indexes = [i * interval for i in range(num_of_required_frames)]
        for index in range(num_of_video_total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if index not in target_indexes:
                continue

            frame = cv2.resize(frame, (FRAME_SHAPE[0], FRAME_SHAPE[1]))
            if FRAME_SHAPE[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame.reshape(FRAME_SHAPE)

            video_frames.append(frame)

        return video_frames

    @staticmethod
    def combine_video_frames(video_frames: list):
        height, width, _ = video_frames[0].shape

        combined_video_frames = np.zeros((height * NUM_OF_FRAMES, width * NUM_OF_FRAMES, FRAME_SHAPE[2]),
                                         dtype=np.uint8)

        for i, frame in enumerate(video_frames):
            r_idx = int(i / NUM_OF_FRAMES)
            c_idx = i % NUM_OF_FRAMES
            combined_video_frames[c_idx * height: (c_idx + 1) * height, r_idx * width: (r_idx + 1) * width, :] = frame

        return combined_video_frames
