import csv
import json
import os
from threading import Lock

import cv2


VIDEO_FPS = 30
AUDIO_FPS = 16000


def read_csv_file(file_path: str, has_header: bool = True):
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            if has_header:
                next(reader, None)

            data = []
            for row in reader:
                if row is not None:
                    data.append(tuple(row))

            return data

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: JSON decoding failed for file - {file_path}")
        return {}
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")
        return {}


def save_json_file(data, output_file_path):
    try:
        with open(output_file_path, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")
        return {}


def read_metadata(metadata_path: str) -> list:
    """
    Read the metadata.json file.
    :param metadata_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\metadata.json
    :return: [["filename.mp4", "REAL" or "FAKE"]]
    """
    metadata = []
    data = read_json_file(metadata_path)
    for filename in data:
        metadata.append((filename, data[filename]["label"]))

    return metadata


def collect_mp4_paths_and_names(root_directory):
    """
    Collects paths and names of all .mp4 files in the given root directory and its subdirectories.

    Parameters:
    root_directory (str): The path to the root directory from which to start searching for .mp4 files.

    Returns:
    list of tuples: A list of tuples where each tuple contains the full path and the file name of an .mp4 file.
    """
    mp4_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                mp4_files.append((full_path, file))
    return mp4_files


def process_batch(frames, resize_shape):
    resized_frames = [cv2.resize(frame, resize_shape) for frame in frames]
    return resized_frames


def read_video_frames(video_path: str):
    """
    Read the video and get the frames.
    :param metadata_path: C:\workspace\deepfake-detection-challenge\train_sample_videos\metadata.json
    :return: the frames from the video
    """
    # print(video_path)
    cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    num_of_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames = []
    for index in range(num_of_video_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, (FRAME_SHAPE[0], FRAME_SHAPE[1]))
        video_frames.append(frame)
    cap.release()

    return video_frames, VIDEO_FPS
