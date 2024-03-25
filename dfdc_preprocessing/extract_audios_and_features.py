import os
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import moviepy.editor as me
import soundfile as sf
import torchaudio
import torchaudio.transforms as T

from dfdc_preprocessing.dfdc_args import get_args
from preprocessing.Preprocessor import collect_mp4_paths_and_names

target_time = 3.6


def run(preprocessed_directory_path, video_path, video_name):
    # Set the target image path
    filename, extension = os.path.splitext(video_name)

    # Extract the wav from the video
    temp_audio_path = f"./temp_{filename}.wav"
    video_clip = me.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(temp_audio_path, codec="pcm_s16le", verbose=False)
    video_clip.close()

    # Crop the wave
    y, sr = librosa.core.load(temp_audio_path, sr=22050)

    target_length = int(sr * target_time)
    remain = len(y) - target_length
    y = y[remain // 2: -(remain - remain // 2)]

    cropped_audio_path = os.path.join(preprocessed_directory_path, f"{filename}_cropped.wav")
    sf.write(cropped_audio_path, y, sr)

    # Remove the temp audio file.
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    # Extract feature
    cropped_feature_path = os.path.join(preprocessed_directory_path, f"{filename}_cropped.png")
    audio, fs = torchaudio.load(cropped_audio_path)
    mfcc_transform = T.MFCC(sample_rate=fs)
    mfcc = mfcc_transform(audio)

    # Create feature image
    plt.clf()
    plt.axis("off")
    plt.imshow(mfcc[0], interpolation="nearest", origin="lower", aspect="auto")
    plt.savefig(cropped_feature_path, bbox_inches='tight', pad_inches=0)


def main():
    print("=" * 40 + " Start preprocess to extract audios" + "=" * 40)
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
