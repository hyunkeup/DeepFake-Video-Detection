from threading import Lock

import cv2
import mediapipe as mp
import moviepy.editor as me
import numpy as np

from property import Property
from utils import FileUtils

FRAME_SHAPE: tuple = tuple(Property.get_property("frame_shape"))
NUM_OF_FRAMES: int = int(Property.get_property("square_of_root_num_of_frames"))

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.8, model_selection=1)

mp_lock = Lock()


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def read_metadata(metadata_path: str) -> list:
        """
        Read the metadata.json file.
        :param metadata_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\metadata.json
        :return: [["filename.mp4", "REAL" or "FAKE"]]
        """
        metadata = []
        data = FileUtils.read_json_file(metadata_path)
        for filename in data:
            metadata.append((filename, data[filename]["label"]))

        return metadata

    @staticmethod
    def read_video_frames(video_path: str):
        """
        Read the video and get the frames.
        :param metadata_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\metadata.json
        :return: the frames from the video
        """
        cap = cv2.VideoCapture(video_path)
        num_of_video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frames = []
        for index in range(num_of_video_total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (FRAME_SHAPE[0], FRAME_SHAPE[1]))
            video_frames.append(frame)

        return video_frames

    @staticmethod
    def combine_video_frames(video_frames: list):
        """
        Combine the video frames. The number frames must be square.
        :param video_frames: the number of frames is there such as 4, 9, 16, 25, ...
        :return: image frame
        """
        height, width, _ = video_frames[0].shape

        combined_video_frames = np.zeros((height * NUM_OF_FRAMES, width * NUM_OF_FRAMES, FRAME_SHAPE[2]),
                                         dtype=np.uint8)

        for i, frame in enumerate(video_frames):
            r_idx = int(i / NUM_OF_FRAMES)
            c_idx = i % NUM_OF_FRAMES
            combined_video_frames[c_idx * height: (c_idx + 1) * height, r_idx * width: (r_idx + 1) * width, :] = frame

        return combined_video_frames

    @staticmethod
    def extract_face(frame):
        """
        Extract the faces from the frame
        :param frame: frame
        :return: frame
        """
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_lock:
            results = face_detection.process(image_rgb)

        if not results.detections:
            return None

        face_images = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)
            face_image = frame[y:y + h, x:x + w]

            if face_image.size != 0:
                face_images.append(face_image)

        if len(face_images) == 0:
            return None

        return face_images

    @staticmethod
    def read_audio_from_video(vide_path: str):
        video = me.VideoFileClip(vide_path)
        audio_array = video.audio.to_soundarray()

        return audio_array
