import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from preprocessing.Preprocessor import Preprocessor
from property import Property
from threading import Lock

# Load .env data
ORIGINAL_HOME_DIRECTORY = Property.get_property("origin_home_directory")
PARTITIONED_DIRECTORIES = Property.get_property("partitioned_directories")
THREAD_POOL_SIZE = Property.get_property("workers_thread_pool_size")
PREPROCESSED_DIRECTORY = Property.get_property("preprocessed_directory")

# test_directory = "C:/workspace/deepfake-detection-challenge/test_videos"
# for file in glob.glob(f"{test_directory}/*.mp4"):
#     video_frames = Preprocessor.read_video_and_extract_face(file)
#     combined_frame = Preprocessor.combine_video_frames(video_frames)
#     filename_with_extension = os.path.basename(file)
#     filename, extension = os.path.splitext(filename_with_extension)
#     combined_image_path = f"C:/workspace/deepfake-detection-challenge/test_images/FAKE/{filename}.jpg"
#     cv2.imwrite(combined_image_path, combined_frame)

lock = Lock()
def run(m_data):
    filename_with_extension, label, path = m_data
    # video_frames = Preprocessor.read_video(f"{path}/{filename_with_extension}")
    with lock:
        video_frames = Preprocessor.read_video_and_extract_face(f"{path}/{filename_with_extension}")
    combined_frame = Preprocessor.combine_video_frames(video_frames)

    filename, extension = os.path.splitext(filename_with_extension)
    combined_image_path = f"{PREPROCESSED_DIRECTORY}/{label}/{filename}.jpg"
    cv2.imwrite(combined_image_path, combined_frame)
    print(f"\t* Saved {combined_image_path}.")


if __name__ == "__main__":
    print("=" * 40 + " Start preprocess " + "=" * 40)
    s_time = time.time()

    # Show current property
    Property.show_property()

    # Create the preprocessed directory
    if not os.path.exists(PREPROCESSED_DIRECTORY):
        print("Create the preprocessed directory: ", end="")
        os.makedirs(PREPROCESSED_DIRECTORY)
        os.makedirs(f"{PREPROCESSED_DIRECTORY}/REAL")
        os.makedirs(f"{PREPROCESSED_DIRECTORY}/FAKE")
        print("Done")
        print(f"\t* {PREPROCESSED_DIRECTORY}")

    # Load metadata
    print("Load metadata: ", end="")
    jobs = []
    for partitioned_directory in PARTITIONED_DIRECTORIES:
        directory_path = f"{ORIGINAL_HOME_DIRECTORY}/{partitioned_directory}"
        metadata_path = f"{directory_path}/metadata.json"
        metadata = Preprocessor.read_metadata(metadata_path)
        jobs.append((directory_path, metadata_path, metadata))
    print("Done")

    # Print jobs
    for _, metadata_path, metadata in jobs:
        print(f"\t* The number of videos: {len(metadata)}, path: {metadata_path}, ")

    # Execute workers
    print("Execute workers: ")
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        for directory_path, _, metadata in jobs:
            for data in metadata:
                executor.submit(run, data + (directory_path,))

    e_time = time.time()
    print("=" * 40 + f" {round(e_time - s_time, 3)} seconds - Done. " + "=" * 40)
