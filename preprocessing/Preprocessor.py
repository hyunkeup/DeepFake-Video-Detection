import os
from threading import Lock
from multiprocessing import Pool
import time

import cv2
import mediapipe as mp
import moviepy.editor as me
import numpy as np
import torchaudio

from property import Property
from utils import FileUtils

FRAME_SHAPE: tuple = tuple(Property.get_property("frame_shape"))
NUM_OF_FRAMES: int = int(Property.get_property("square_of_root_num_of_frames"))
ORIGINAL_HOME_DIRECTORY = Property.get_property("origin_home_directory")
PARTITIONED_DIRECTORIES = Property.get_property("partitioned_directories")
PREPROCESSED_DIRECTORY = Property.get_property("preprocessed_directory")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.8, model_selection=1)

mp_lock = Lock()

VIDEO_FPS = 30
AUDIO_FPS = 16000


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
    def process_batch(frames, resize_shape):
        resized_frames = [
            cv2.resize(frame, resize_shape) for frame in frames
        ]
        return resized_frames

    @staticmethod
    def read_video_frames(video_path: str):
        """
        Read the video and get the frames.
        :param metadata_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\metadata.json
        :return: the frames from the video
        """
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        num_of_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frames = []
        for index in range(num_of_video_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (FRAME_SHAPE[0], FRAME_SHAPE[1]))
            video_frames.append(frame)
        cap.release()

        return video_frames, VIDEO_FPS

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
    def read_audio_from_video(video_path: str):
        # Extract the audio file from the video
        filename, extension = os.path.splitext(os.path.basename(video_path))
        audio_path = f"./temp_{filename}.wav"
        video_clip = me.VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, codec='pcm_s16le')
        video_clip.close()

        # Resample the audio 44100 -> 16000
        audio_frames, origin_audio_fps = torchaudio.load(audio_path)
        audio_frames = torchaudio.functional.resample(audio_frames, orig_freq=origin_audio_fps, new_freq=AUDIO_FPS)

        # Remove the temp audio file.
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return audio_frames.T, AUDIO_FPS
    

    @staticmethod
    def process_video(directory_path, filename_with_extension):
        video_path = os.path.join(directory_path, filename_with_extension)
        video_frames, video_fps = Preprocessor.read_video_frames(video_path)
        audio_frames, audio_fps = Preprocessor.read_audio_from_video(video_path)

        video_chunk_size = int(video_fps * 0.2)
        audio_chunk_size = int(audio_fps * 0.2)
        num_of_units = int(len(video_frames) / video_chunk_size)

        return [(video_frames[i * video_chunk_size: (i + 1) * video_chunk_size],
                 audio_frames[i * audio_chunk_size: (i + 1) * audio_chunk_size])
                for i in range(num_of_units)]

    @staticmethod
    def load_dataset():
        start = time.time()
        print("Load metadata: ", end="")
        jobs = []
        for partitioned_directory in PARTITIONED_DIRECTORIES:
            directory_path = f"{ORIGINAL_HOME_DIRECTORY}/{partitioned_directory}"
            metadata_path = f"{directory_path}/metadata.json"
            metadata = Preprocessor.read_metadata(metadata_path)
            for filename_with_extension, label in metadata:
                jobs.append((directory_path, filename_with_extension))

        print("Done")

        with Pool(4) as pool:
            results = pool.starmap(Preprocessor.process_video, jobs)

        dataset = [item for sublist in results for item in sublist]

        print(f"Finished preprocessing: in {time.time() - start} seconds")
        return dataset


