import csv
import glob
import json
import os

import cv2

deepfack_detection_challenge_home = "C:/workspace/deepfake-detection-challenge"
train_sample_videos_path = f"{deepfack_detection_challenge_home}/train_sample_videos"
train_sample_videos_metadata_path = f"{train_sample_videos_path}/metadata.json"
test_videos_path = f"{deepfack_detection_challenge_home}/test_videos"
test_videos_path_metadata_path = f"{deepfack_detection_challenge_home}/sample_submission.csv"


def read_csv_file(file_path):
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)

            data = []
            for row in reader:
                if row is not None:
                    data.append(tuple(row))

            return data

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except json.JSONDecodeError:
        print(f"Error: JSON decoding failed for file - {file_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")


# Get train video files
def get_train_video_filepath():
    return glob.glob(f"{train_sample_videos_path}/*.mp4")


def get_train_video_metadata():
    result = []
    data = read_json_file(train_sample_videos_metadata_path)
    for filename in data:
        result.append((filename, "0" if data[filename]["label"] == "FAKE" else "1"))

    return result

def get_test_video_filepath():
    return glob.glob(f"{test_videos_path}/*.mp4")


def get_test_video_metadata():
    return read_csv_file(test_videos_path_metadata_path)


def video_to_images(video_path, output_folder, resize_shape=None):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_folder, exist_ok=True)

    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if resize_shape:
            frame = cv2.resize(frame, resize_shape)

        image_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(image_filename, frame)

    cap.release()


def print_video_data_info():
    print("=" * 20 + " Train datasets " + "=" * 20)
    files = get_train_video_filepath()
    print(f"Num of files: {len(files)}")
    print(f"Files: {files}")

    video_path = files[0]
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Total_frames: {total_frames}")
    print(f"(width, height): ({width}, {height})")

    metadata = get_train_video_metadata()
    print(metadata)

    print("=" * 20 + " Test datasets " + "=" * 20)
    files = get_test_video_filepath()
    print(f"Num of files: {len(files)}")
    print(f"Files: {files}")

    video_path = files[0]
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Total_frames: {total_frames}")
    print(f"(width, height): ({width}, {height})")

    metadata = get_test_video_metadata()
    print(metadata)


if __name__ == "__main__":
    print_video_data_info()
