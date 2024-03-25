import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from dfdc_preprocessing.dfdc_args import get_args
from preprocessing.Preprocessor import collect_mp4_paths_and_names, read_video_frames

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=(1080, 1920), device=device)

save_frames = 15
input_fps = 30

save_length = 3.6
save_avi = False


def get_select_distribution(m, n):
    return [i * n // m + n // (2 * m) for i in range(m)]


def run(preprocessed_directory_path, video_path, video_name):
    try:
        # Set the target image path
        filename, extension = os.path.splitext(video_name)

        # Get frames
        frames, fps = read_video_frames(video_path)

        # Get target frames
        frame_n = int(save_length * input_fps)
        selected_frames = get_select_distribution(save_frames, frame_n)

        face_frames = []
        target_frames = [frames[i] for i in selected_frames]
        for frame in target_frames:
            # Extract a face on the video.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox = mtcnn.detect(image_rgb)
            if bbox[0] is not None:
                bbox = bbox[0][0]
                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox
            try:
                frame = frame[y1:y2, x1:x2, :]
                frame = cv2.resize(frame, (224, 224))
                face_frames.append(frame)
            except UnboundLocalError as e:
                face_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # Save the cropped video if you want.
        if save_avi:
            save_fps = save_frames // (frame_n // input_fps)
            cropped_video_path = os.path.join(
                PREPROCESSED_DIRECTORY, f"{filename}_face_cropped.avi"
            )
            out = cv2.VideoWriter(
                cropped_video_path,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                save_fps,
                (224, 224),
            )

            for face_frame in face_frames:
                out.write(face_frame)
            out.release()

        # Save the numpy data for training.
        cropped_npfile_path = os.path.join(preprocessed_directory_path, f"{filename}_face_cropped.npy")
        np.save(cropped_npfile_path, np.array(face_frames))
        print(f"\t* Saved {cropped_npfile_path} from '{os.path.basename(video_path)}'.")

    except Exception as e:
        print("ERROR:", e)


def main():
    print("=" * 40 + " Start preprocess to extract faces " + "=" * 40)
    s_time = time.time()

    # Create the preprocessed directory
    for split_directory in ["real", "fake"]:
        preprocessed_directory_path = os.path.join(PREPROCESSED_DIRECTORY, split_directory)

        if not os.path.exists(preprocessed_directory_path):
            print("Create the preprocessed directory: ")
            os.makedirs(preprocessed_directory_path)
            print(f"\t* {preprocessed_directory_path}")

    # Execute workers
    print("Execute workers: ")
    for split_directory in ["real", "fake"]:
        preprocessed_directory_path = os.path.join(PREPROCESSED_DIRECTORY, split_directory)
        origin_directory_path = os.path.join(ORIGINAL_HOME_DIRECTORY, split_directory)
        candidates = collect_mp4_paths_and_names(origin_directory_path)

        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
            for video_path, video_name in candidates:
                executor.submit(run, preprocessed_directory_path, video_path, video_name)

    e_time = time.time()
    print("=" * 40 + f" {round(e_time - s_time, 3)} seconds - Done. " + "=" * 40)


if __name__ == "__main__":
    args = get_args()

    ORIGINAL_HOME_DIRECTORY = args.root_dir
    THREAD_POOL_SIZE = args.num_threads
    PREPROCESSED_DIRECTORY = args.save_dir

    main()
