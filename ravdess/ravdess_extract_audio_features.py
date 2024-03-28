import glob
import os
import time

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from tqdm import tqdm


def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc


def audio_load(file_path):
    '''
    Loads and resample audio if not at 44100
    '''
    target_sr = 44100
    try:
        audio_data, sr = librosa.load(file_path, sr=None, mono=True)
        # check sampling rate is 44100Hz
        if sr != target_sr:
            # resample audio to 44100 (easier to work with to match with video frames)
            num_samples = len(audio_data)
            audio_data = resample(audio_data, int(num_samples * target_sr / sr))
        return audio_data, target_sr  # resampled audio data, target sampling rate
    except Exception as e:
        print(f"Error: {e}")


def get_ravdess_metadata(dir_path):
    # https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data
    """
        0: Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
        1: Vocal channel (01 = speech, 02 = song).
        2*: Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        3: Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        4: Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        5: Repetition (01 = 1st repetition, 02 = 2nd repetition).
        6: Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    :param dir_path:
    :return:
    """
    metadata = []
    for wav_file in glob.glob(os.path.join(dir_path, "*.wav")):
        directory_path = os.path.dirname(wav_file)
        filename = os.path.basename(wav_file)
        filename_without_extension = filename.split(".")[0]

        # emotion index is 2
        data = {
            "directory_path": directory_path,
            "filename": filename,
            "label": filename.split(".")[0].split("-")[2],
            "wav_file_path": wav_file,
            "image_file_path": os.path.join(directory_path, filename_without_extension + ".png"),
            "np_file_path": os.path.join(directory_path, filename_without_extension + ".npy")
        }
        metadata.append(data)

    return metadata


def generate_datasets(metadata, min_duration=3):
    print("=" * 50 + " Generating datasets " + "=" * 50)
    s_time = time.time()
    for data in tqdm(metadata):
        wav_file_path = data["wav_file_path"]
        image_file_path = data["image_file_path"]
        np_file_path = data["np_file_path"]

        # Load the audio file
        audio, sr = audio_load(wav_file_path)  # 3 seconds

        # Padding or Cutting by min_duration
        audio_length = min_duration * sr
        if audio.shape[0] < min_duration * sr:
            # Padding
            audio = np.pad(audio, (0, audio_length - len(audio)), 'constant')
        elif audio.shape[0] > audio_length:
            # Cutting
            audio = audio[:audio_length]

        audio_frames = librosa.util.frame(audio, frame_length=sr, hop_length=sr)
        audio_frames = np.transpose(audio_frames)
        assert audio_frames.shape[0] == min_duration, f"The audio frames should have {min_duration} seconds duration."

        # Extract audio features
        audio_features = [get_mfccs(y=audio_frame, sr=sr) for audio_frame in audio_frames]

        # Plotting MFCC features
        plt.clf()
        plt.axis("off")
        plt.imshow(np.vstack(audio_features), interpolation="nearest", origin="lower", aspect="auto")
        plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)

        # Resize for 244 x 244
        frame = cv2.imread(image_file_path)
        frame = cv2.resize(frame, (244, 244))
        cv2.imwrite(image_file_path, frame)
        np.save(np_file_path, np.array(frame))

    print("=" * 50 + " Datasets are generated. " + "=" * 50)
    print(f"{round(time.time() - s_time, 2)} seconds.")


def main():
    dir_path = "C:\\workspace\\deepfake-detection-challenge\\audio_resampled"
    metadata = get_ravdess_metadata(dir_path)
    generate_datasets(metadata)


if __name__ == "__main__":
    main()
