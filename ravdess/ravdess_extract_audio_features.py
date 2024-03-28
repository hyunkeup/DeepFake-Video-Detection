import glob
import os

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample


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
    for file_path in glob.glob(os.path.join(dir_path, "*.wav")):
        directory_path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        # emotion index is 2
        data = {
            "directory_path": directory_path,
            "filename": filename,
            "path": file_path,
            "label": filename.split(".")[0].split("-")[2]
        }
        metadata.append(data)

    return metadata


def generate_datasets(metadata, min_duration=3):
    for data in metadata:
        directory_path = data["directory_path"]
        filename = data["filename"]
        path = data["path"]
        label = data["label"]
        filename_without_extension = filename.split(".")[0]

        # Load the audio file
        audio, sr = audio_load(path)  # 3 seconds

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
        mfcc_img_path = os.path.join(directory_path, filename_without_extension + ".png")
        plt.clf()
        plt.axis("off")
        plt.imshow(np.vstack(audio_features), interpolation="nearest", origin="lower", aspect="auto")
        plt.savefig(mfcc_img_path, bbox_inches='tight', pad_inches=0)

        # Resize for 244 x 244
        frame = cv2.imread(mfcc_img_path)
        frame = cv2.resize(frame, (244, 244))
        cv2.imwrite(mfcc_img_path, frame)
        np.save(os.path.join(directory_path, filename_without_extension + ".npy"), np.array(frame))


def main():
    dir_path = "C:\\workspace\\deepfake-detection-challenge\\audio_resampled"
    metadata = get_ravdess_metadata(dir_path)
    generate_datasets(metadata)


if __name__ == "__main__":
    main()
