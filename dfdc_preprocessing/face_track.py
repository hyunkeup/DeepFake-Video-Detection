import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

from dfdc_preprocessing.dfdc_args import get_args


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


def read_json_file(file_path):
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def save_json_file(data, output_file_path):
    """Saves the modified data to a JSON file."""
    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4)


def face_detect_single_video(path, video_name, mtcnn, num_frames_to_process):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        new_frame = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frames.append(new_frame)

    if len(frames) == 0:
        return video_name, -1

    # Ensure not to exceed the list length when randomly selecting frames
    num_frames_to_process = min(num_frames_to_process, len(frames))

    # Randomly select frames
    selected_frames = random.sample(frames, num_frames_to_process)
    try:
        max_faces = 0
        for frame in selected_frames:
            boxes, _ = mtcnn.detect(frame)
            max_faces = max(max_faces, len(boxes) if boxes is not None else 0)
        return max_faces

    except Exception as e:
        print("ERROR:", e, "file: ", video_name)
        return -1


def face_detect(video_paths, video_dict, root_dict, input_metadata_filename,
                final_output_filename="final_metadata.json", num_frames_to_process=3, num_threads=1):
    """Detect people amount in the video."""

    # Check if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load current meta json
    mtcnn = MTCNN(keep_all=True, device=device)
    input_json_data = read_json_file(os.path.join(video_dict, input_metadata_filename))

    # Load or initialize the final consolidated metadata
    final_json_path = os.path.join(root_dict, final_output_filename)
    final_json_data = read_json_file(final_json_path)

    if num_threads > 1:
        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    face_detect_single_video, path, video_name, mtcnn, num_frames_to_process
                )
                for path, video_name in video_paths
            ]
            for future in as_completed(futures):
                video_name, max_faces = future.result()
                final_json_data[video_name] = input_json_data.get(video_name, {})
                final_json_data[video_name]["speaker_count"] = max_faces
                print(f"Processed video: {video_name} has {max_faces} people.")
    else:
        # Process sequentially
        print(f"Processing {video_dict}")
        for path, video_name in tqdm(video_paths):
            max_faces = face_detect_single_video(path, video_name, mtcnn, num_frames_to_process)
            final_json_data[video_name] = input_json_data.get(video_name, {})
            final_json_data[video_name]["speaker_count"] = max_faces

    # Save the updated data back to the final metadata file
    save_json_file(final_json_data, final_json_path)


if __name__ == "__main__":
    args = get_args()
    root_dir = args.root_dir
    sub_folders = args.sub_folders
    input_json = "metadata.json"
    final_json = "final_metadata.json"

    num_threads = args.num_threads

    for sub in sub_folders:
        video_dir = os.path.join(root_dir, sub)
        candidates = collect_mp4_paths_and_names(video_dir)
        face_detect(candidates, video_dir, root_dir, input_json, final_json, num_threads=num_threads)
        print(f"Output new metadata file from {video_dir}")
