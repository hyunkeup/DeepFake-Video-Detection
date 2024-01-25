import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from preprocessing.Preprocessor import Preprocessor
from property import Property

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

def run(m_data):
    # Set the target image path
    filename_with_extension, label, directory_path = m_data
    filename, extension = os.path.splitext(filename_with_extension)
    combined_image_path = f"{PREPROCESSED_DIRECTORY}/{label}/{filename}.jpg"
    if os.path.exists(combined_image_path):
        print(f"\t* Skipped {combined_image_path} from '{os.path.basename(directory_path)}'.")
        return

    # Get frames
    video_frames = Preprocessor.read_video(f"{directory_path}/{filename_with_extension}")
    # Save the images
    for i, frame in enumerate(video_frames):
        image_path = f"{PREPROCESSED_DIRECTORY}/{label}/{filename}_{i}.jpg"
        cv2.imwrite(image_path, frame)
        print(f"\t* Saved {image_path} from '{os.path.basename(directory_path)}'.")

    # # Get face frames
    # video_frames = Preprocessor.read_video_and_extract_face2(f"{directory_path}/{filename_with_extension}")
    #
    # # Combine the frames
    # combined_frame = Preprocessor.combine_video_frames(video_frames)
    #
    # # Save the combined image
    # cv2.imwrite(combined_image_path, combined_frame)
    # print(f"\t* Saved {combined_image_path} from '{os.path.basename(directory_path)}'.")


def main():
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