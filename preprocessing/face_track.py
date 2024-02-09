import os
import time
import torch
import json
import mmcv, cv2
from facenet_pytorch import MTCNN
from PIL import Image


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
    with open(file_path, "r") as file:
        return json.load(file)


def save_json_file(data, output_file_path):
    """Saves the modified data to a JSON file."""
    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4)


def face_detect(video_paths, root):
    """Detect people amount in the video."""

    # Check if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model and load meta json
    mtcnn = MTCNN(keep_all=True, device=device)
    json_data = read_json_file(root + "metadata.json")

    # Detection
    for i, (path, video_name) in enumerate(video_paths):
        # Read video
        video = mmcv.VideoReader(path)
        frames = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video
        ]
        if not frames:
            continue
        try:
            # Face detection and track the max face amount detected only for certain frames
            frames_count = 1
            max_faces = 0
            for frame in frames[:frames_count]:
                # Detect faces
                boxes, _ = mtcnn.detect(frame)
                # Update faces count
                max_faces = max(max_faces, len(boxes))

            json_data[video_name]["speaker_count"] = max_faces
            print(
                f"Processing {i + 1}/{len(candidates)} videos: {video_name} with {max_faces} people."
            )
        except Exception as e:
            print("ERROR:", e, "file: ", video_name)

    save_json_file(json_data, root + "new_metadata.json")
    return


if __name__ == "__main__":
    s_time = time.time()
    root_dict = "../dataset/test_videos/"
    candidates = collect_mp4_paths_and_names(root_dict)
    face_detect(candidates, root_dict)
    e_time = time.time()
    print(f"Output new metadata file in {root_dict}")
    print(f" Took {round((e_time - s_time) / 60, 3)} mins. ")
