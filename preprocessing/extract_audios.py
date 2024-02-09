import os
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import moviepy.editor as me
import soundfile as sf

from preprocessing.Preprocessor import Preprocessor
from property import Property

# Load property
ORIGINAL_HOME_DIRECTORY = Property.get_property("origin_home_directory")
PARTITIONED_DIRECTORIES = Property.get_property("partitioned_directories")
THREAD_POOL_SIZE = Property.get_property("workers_thread_pool_size")
PREPROCESSED_DIRECTORY = Property.get_property("preprocessed_directory")

target_time = 3.6


def run(m_data):
    # Set the target image path
    filename_with_extension, label, directory_path = m_data
    filename, extension = os.path.splitext(filename_with_extension)

    # Extract the wav from the video
    video_path = os.path.join(directory_path, filename_with_extension)
    temp_audio_path = f"./temp_{filename}.wav"
    video_clip = me.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le')
    video_clip.close()

    # Crop the wave
    y, sr = librosa.core.load(temp_audio_path, sr=22050)

    target_length = int(sr * target_time)
    remain = len(y) - target_length
    y = y[remain // 2:-(remain - remain // 2)]

    sf.write(os.path.join(PREPROCESSED_DIRECTORY, f"{filename}_cropped.wav"), y, sr)

    # Remove the temp audio file.
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)


def main():
    print("=" * 40 + " Start preprocess to extract audios" + "=" * 40)
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
