import shutil
from moviepy.editor import *
from tqdm import tqdm
import json

from dfdc_preprocessing.face_track import collect_mp4_paths_and_names, read_json_file


# Function to calculate label ratio
def calculate_label_ratio(data):
    # Initialize counters for REAL and FAKE labels
    real_count = 0
    fake_count = 0

    # Iterate over each item in the dictionary to count the labels
    for key, value in data.items():
        if value["label"] == "REAL":
            real_count += 1
        elif value["label"] == "FAKE":
            fake_count += 1

    # Calculate the ratio of REAL to FAKE
    total = real_count + fake_count
    if total > 0:  # To avoid division by zero
        real_ratio = real_count / total
        fake_ratio = fake_count / total
    else:
        real_ratio = fake_ratio = 0

    return real_ratio, fake_ratio, real_count

def extract_audio_by_video(video_path, video_name, save_root):
    try:
        # Specify the output path for the .wav file
        audio_output_path = video_name.split('.')[0] + '.wav'
        audio_output_path = os.path.join(save_root, audio_output_path)

        # Load the video file
        video = VideoFileClip(video_path)

        # Extract the audio from the video
        audio = video.audio

        # Save the audio as a .wav file
        audio.write_audiofile(audio_output_path, verbose=False, logger=None)

        # Close the video file to release resources
        video.close()

        return audio_output_path

    except Exception as e:
        print(f"Error extracting audio: {e}, file name: {video_name}")
        return "no_audio"


def move_file(src_file_path, des_dir_path, sub_folder="train"):
    # Check data type is for training or not
    des_dir_path = os.path.join(des_dir_path, sub_folder)

    try:
        # Check if the destination directory exists, create if not
        if not os.path.exists(des_dir_path):
            os.makedirs(des_dir_path)

        # Extract the base name of the source file to append to the destination directory path
        file_name = os.path.basename(src_file_path)
        # Construct the full destination file path
        des_file_path = os.path.join(des_dir_path, file_name)
        # Move the file
        shutil.move(src_file_path, des_file_path)
        return True

    except Exception as e:
        print(f"Error moving file: {e}")
        return False


if __name__ == "__main__":
    des_dir_path = "../datasets/raw_dataset/raw_video_audio_pairs/"
    video_folder = "../datasets/raw_dataset/raw_videos/"
    candidates = collect_mp4_paths_and_names(video_folder)
    meta_json_name = "final_metadata.json"
    json_data = read_json_file(os.path.join(video_folder, meta_json_name))

    for video_path, video_name in tqdm(candidates):
        if json_data[video_name]["speaker_count"] == 1:
            audio_output_path = extract_audio_by_video(video_path=video_path,
                                                       video_name=video_name,
                                                       save_root=video_folder)
            if audio_output_path == "no_audio":
                continue
            sub_folder_name = None
            if json_data[video_name]["label"] == "REAL":
                sub_folder_name = "real"
            else:
                sub_folder_name = "fake"
            move_file(video_path, des_dir_path, sub_folder=sub_folder_name)
            move_file(audio_output_path, des_dir_path, sub_folder=sub_folder_name)


    # # Check dataset balanced
    # json_data = read_json_file("../datasets/final_metadata.json")
    # real_ratio, fake_ratio, real_count = calculate_label_ratio(json_data)
    # print(real_ratio, fake_ratio, real_count)

