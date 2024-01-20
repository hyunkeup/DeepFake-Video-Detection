import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import dotenv

from preprocessing.Preprocessor import Preprocessor

# Load .env data
dotenv.load_dotenv()
ORIGINAL_DIRECTORY = os.environ.get("ORIGINAL_DIRECTORY")
THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE"))
PREPROCESSED_DIRECTORY = os.environ.get("PREPROCESSED_DIRECTORY")

preprocessor = Preprocessor()


def run(m_data):
    filename_with_extension, label = m_data
    video_frames = preprocessor.read_video(f"{ORIGINAL_DIRECTORY}/{filename_with_extension}")
    combined_frame = preprocessor.combine_video_frames(video_frames)

    filename, extension = os.path.splitext(filename_with_extension)
    combined_image_path = f"{PREPROCESSED_DIRECTORY}/{label}/{filename}.jpg"
    cv2.imwrite(combined_image_path, combined_frame)
    print(f"\t* Saved {combined_image_path}.")


if __name__ == "__main__":
    print("=" * 40 + " Start preprocess " + "=" * 40)
    s_time = time.time()

    # Load metadata
    print("Load metadata: ", end="")
    origin_metadata_path = f"{ORIGINAL_DIRECTORY}/metadata.json"
    metadata = preprocessor.read_metadata(origin_metadata_path)
    print("Done")
    print(f"\t* Path: {origin_metadata_path}")
    print(f"\t* The number of videos: {len(metadata)}")

    # Remove the previous data
    if os.path.exists(PREPROCESSED_DIRECTORY):
        shutil.rmtree(PREPROCESSED_DIRECTORY)

    # Create the preprocessed directory
    print("Create the preprocessed directory: ", end="")
    os.makedirs(PREPROCESSED_DIRECTORY)
    os.makedirs(f"{PREPROCESSED_DIRECTORY}/REAL")
    os.makedirs(f"{PREPROCESSED_DIRECTORY}/FAKE")
    print("Done")
    print(f"\t* {PREPROCESSED_DIRECTORY}")

    # Execute
    print("Execute workers: ")
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        for data in metadata:
            executor.submit(run, data)

    e_time = time.time()
    print("=" * 40 + f" {round(e_time - s_time, 3)} seconds - Done. " + "=" * 40)
