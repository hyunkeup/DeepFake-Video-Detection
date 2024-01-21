import cv2
import mediapipe as mp
import numpy as np

from property import Property
from utils import FileUtils

FRAME_SHAPE: tuple = tuple(Property.get_property("frame_shape"))
NUM_OF_FRAMES: int = int(Property.get_property("square_of_root_num_of_frames"))

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)


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
    def combine_video_frames(video_frames: list):
        """
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
    def read_video(video_path: str) -> list:
        """
        :param video_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\aaa.mp4
        :return: image frames
        """
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
            video_frames.append(frame)

        return video_frames

    @staticmethod
    def extract_face(frame):
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if not results.detections:
            return None

        face_images = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)
            face_image = frame[y:y + h, x:x + w]
            face_images.append(face_image)

        return face_images

    @staticmethod
    def read_video_and_extract_face(video_path: str):
        """
        :param video_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\aaa.mp4
        :return: image frames
        """
        cap = cv2.VideoCapture(video_path)
        num_of_video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_of_required_frames = NUM_OF_FRAMES ** 2
        if num_of_video_total_frames < num_of_required_frames:
            raise Exception(
                f"The num_of_video_total_frames must be greater than num_of_required_frames. num_of_video_total_frames: {num_of_video_total_frames}, num_of_required_frames: {NUM_OF_FRAMES}")

        extracted_face_frames = []
        for index in range(num_of_video_total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            face_frames = Preprocessor.extract_face(frame)
            if face_frames is None:
                continue

            for face_image in face_frames:
                face_image = cv2.resize(face_image, (FRAME_SHAPE[0], FRAME_SHAPE[1]))
                extracted_face_frames.append(face_image)
        extracted_face_frames = np.array(extracted_face_frames)

        interval = len(extracted_face_frames) // num_of_required_frames
        target_indexes = [i * interval for i in range(num_of_required_frames)]

        return extracted_face_frames[target_indexes]
