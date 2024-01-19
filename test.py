from preprocessing.KaggleDatasetLoader import KaggleDatasetLoader


if __name__ == "__main__":
    loader = KaggleDatasetLoader()
    loader.load(directory_path="C:/workspace/deepfake-detection-challenge/train_sample_videos")