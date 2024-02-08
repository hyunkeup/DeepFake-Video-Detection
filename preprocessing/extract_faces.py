import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from preprocessing.Preprocessor import Preprocessor
from property import Property

# Load .env data
ORIGINAL_HOME_DIRECTORY = Property.get_property("origin_home_directory")
PARTITIONED_DIRECTORIES = Property.get_property("partitioned_directories")
THREAD_POOL_SIZE = Property.get_property("workers_thread_pool_size")
PREPROCESSED_DIRECTORY = Property.get_property("preprocessed_directory")

save_frames = 15
input_fps = 30

save_length = 3.6
save_avi = False


def get_select_distribution(m, n):
    return [i * n // m + n // (2 * m) for i in range(m)]


def run(m_data):
    # Set the target image path
    filename_with_extension, label, directory_path = m_data
    filename, extension = os.path.splitext(filename_with_extension)

    # Get frames
    frames, fps = Preprocessor.read_video_frames(os.path.join(directory_path, filename_with_extension))

    # Get target frames
    frame_n = int(save_length * input_fps)
    selected_frames = get_select_distribution(save_frames, frame_n)

    face_frames = []
    target_frames = [frames[i] for i in selected_frames]
    for frame in target_frames:
        face_frame = Preprocessor.extract_face(frame)[0]
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frames.append(face_frame)

    # Save the cropped video if you want.
    if save_avi:
        save_fps = save_frames // (frame_n // input_fps)
        cropped_video_path = os.path.join(PREPROCESSED_DIRECTORY, f"{filename}_face_cropped.avi")
        out = cv2.VideoWriter(cropped_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

        for face_frame in face_frames:
            out.write(face_frame)
        out.release()

    # Save the numpy data for training.
    cropped_npfile_path = os.path.join(PREPROCESSED_DIRECTORY, f"{filename}_face_cropped.npy")
    np.save(os.path.join(PREPROCESSED_DIRECTORY, cropped_npfile_path), np.array(face_frames))
    print(f"\t* Saved {cropped_npfile_path} from '{os.path.basename(directory_path)}'.")


def main():
    print("=" * 40 + " Start preprocess to extract faces " + "=" * 40)
    s_time = time.time()

    # Show current property
    Property.show_property()

    # Create the preprocessed directory
    if not os.path.exists(PREPROCESSED_DIRECTORY):
        print("Create the preprocessed directory: ", end="")
        os.makedirs(PREPROCESSED_DIRECTORY)
        print("Done")
        print(f"\t* {PREPROCESSED_DIRECTORY}")

    # Load metadata
    print("Load metadata: ", end="")
    jobs = []
    for partitioned_directory in PARTITIONED_DIRECTORIES:
        directory_path = os.path.join(ORIGINAL_HOME_DIRECTORY, partitioned_directory)
        metadata_path = os.path.join(directory_path, "metadata.json")
        metadata = Preprocessor.read_metadata(metadata_path)
        jobs.append((directory_path, metadata_path, metadata))
    print("Done")

    # Print jobs
    for _, metadata_path, metadata in jobs:
        print(f"\t* The number of videos: {len(metadata)}, path: {metadata_path}.")

    # Execute workers
    print("Execute workers: ")
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        for directory_path, _, metadata in jobs:
            for data in metadata:
                executor.submit(run, data + (directory_path,))

    e_time = time.time()
    print("=" * 40 + f" {round(e_time - s_time, 3)} seconds - Done. " + "=" * 40)


if __name__ == "__main__":
    main()
