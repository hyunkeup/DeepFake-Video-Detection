from preprocessing.DfdcDatasetsLoader import DfdcDatasetsLoader


if __name__ == "__main__":
    loader = DfdcDatasetsLoader()
    loader.load(directory_path="C:/workspace/deepfake-detection-challenge/train_sample_videos")