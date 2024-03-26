import shutil
from moviepy.editor import *
from tqdm import tqdm

from preprocessing.face_track import collect_mp4_paths_and_names, read_json_file


def extract_audio_by_video(video_path, video_name, save_root):
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


def move_file(src_file_path, dest_dir_path, type="train"):
    # Check data type is for training or not
    if type == "train":
        dest_dir_path = os.path.join(dest_dir_path, "train")
    else:
        dest_dir_path = os.path.join(dest_dir_path, "test")

    try:
        # Check if the destination directory exists, create if not
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)

        # Extract the base name of the source file to append to the destination directory path
        file_name = os.path.basename(src_file_path)
        # Construct the full destination file path
        dest_file_path = os.path.join(dest_dir_path, file_name)
        # Move the file
        shutil.move(src_file_path, dest_file_path)
        return True

    except Exception as e:
        print(f"Error moving file: {e}")
        return False


if __name__ == "__main__":
    raw_dataset_dir = "../dataset/raw_dataset"
    video_folder = "../dataset/test_videos/total_video"
    candidates = collect_mp4_paths_and_names(video_folder)
    meta_json_name = "final_metadata.json"
    json_data = read_json_file(os.path.join(video_folder, meta_json_name))

    for video_path, video_name in tqdm(candidates):
        if json_data[video_name]["speaker_count"] == 1:
            audio_output_path = extract_audio_by_video(video_path=video_path, video_name=video_name, save_root=video_folder)
            if json_data[video_name]["split"] == "train":
                move_file(video_path, raw_dataset_dir, type="train")
                move_file(audio_output_path, raw_dataset_dir, type="train")
            else:
                move_file(video_path, raw_dataset_dir, type="test")
                move_file(audio_output_path, raw_dataset_dir, type="test")
