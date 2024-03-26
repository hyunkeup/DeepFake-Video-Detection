import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from facenet_pytorch import MTCNN

from dfdc_preprocessing.dfdc_args import get_args
from preprocessing import Preprocessor


def face_detect_single_video(path, video_name, mtcnn, num_frames_to_process):
    frames, _ = Preprocessor.read_video_frames(path)

    if len(frames) == 0:
        return video_name, -1

    # Ensure not to exceed the list length when randomly selecting frames
    num_frames_to_process = min(num_frames_to_process, len(frames))

    # Randomly select frames
    selected_frames = random.sample(frames, num_frames_to_process)
    try:
        max_faces = 0
        for frame in selected_frames:
            boxes, probs = mtcnn.detect(frame)
            if boxes is None:
                continue
            boxes = boxes[probs > 0.9]
            max_faces = max(max_faces, len(boxes) if boxes is not None else 0)

        print(f"Processed video: {video_name} has {max_faces} people.")
        return video_name, max_faces

    except Exception as e:
        print("ERROR:", e, "file: ", video_name)
        return video_name, -1


def face_detect(video_paths, video_dict, input_metadata_filename,
                final_output_filename="final_metadata.json", num_frames_to_process=3, num_threads=1):
    """Detect people amount in the video."""

    # Check if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load current meta json
    mtcnn = MTCNN(keep_all=True, device=device)
    input_json_data = Preprocessor.read_json_file(os.path.join(video_dict, input_metadata_filename))

    # Load or initialize the final consolidated metadata
    final_json_data = input_json_data.copy()

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
            if max_faces == -1:
                print(f"Not processed video: {video_name} has {max_faces}")
                continue
            final_json_data[video_name] = input_json_data.get(video_name, {})
            final_json_data[video_name]["speaker_count"] = max_faces

        # Save the updated data back to the final metadata file
        final_json_path = os.path.join(video_dict, final_output_filename)
        Preprocessor.save_json_file(final_json_data, final_json_path)


def main():
    args = get_args()
    root_dir = args.root_dir
    sub_folders = args.sub_folders
    input_json = "metadata.json"
    final_json = "final_metadata.json"

    num_threads = args.num_threads

    for sub in sub_folders:
        video_dir = os.path.join(root_dir, sub)
        candidates = Preprocessor.collect_mp4_paths_and_names(video_dir)
        face_detect(candidates, video_dir, input_json, final_json, num_threads=num_threads)
        print(f"Output new metadata file from {video_dir}")


if __name__ == "__main__":
    main()
