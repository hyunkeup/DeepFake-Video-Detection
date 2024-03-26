# DeepFake Detection

## Install

```
# Pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Others
pip install numpy opencv-python moviepy librosa facenet_pytorch matplotlib
```

## Dataset Preprocessing

The implementation includes several preprocessing steps to prepare the DFDC dataset for training:

1. **Detect People Count in Videos**: We currently focus on videos featuring a single person. The detection results are
   saved in `final_metadata.json`.
    -
   Script: `dfdc_preprocessing/speaker_labeling.py --root_dir {./full_dataset} --sub_folders {dfdc_train_part_0 dfdc_train_part_1...} --num_threads {1}`

2. **Extract Audio from Video**: We separate the audio component from the raw video samples.
   -
   Script: `dfdc_preprocessing/prepare_raw_dataset.py --root_dir {./full_dataset} --sub_folders {dfdc_train_part_0 dfdc_train_part_1...} --save_dir {./raw_dataset}`

3. **Extract Cropped Face Segments**: Faces are cropped from the raw video and saved as `.npy` files for training.
   -
   Script: `dfdc_preprocessing/extract_faces.py --root_dir {./prepared_raw_sample_videos} --save_dir {./preprocessed_sample_videos} --num_threads {4}`

4. **Extract Cropped Audio**: Corresponding audio segments are extracted from the cropped videos and saved as `.wav`
   and `.png` files.
    -
   Script: `dfdc_preprocessing/extract_audios_and_features.py -root_dir {./prepared_raw_sample_videos} --save_dir {./preprocessed_sample_videos} --num_threads {4}`

5. **Create Annotations**: We generate `{--root_dir}/annotation.txt` files from the processed dataset for use
   with PyTorch's DataLoader.
    - Script: `dfdc_preprocessing/create_annotations.py --root_dir {./prepared_raw_sample_videos}`

## Original Dataset information from DFDC

[Deepfake Detection Challenge Dataset](https://www.kaggle.com/c/deepfake-detection-challenge)

* Each video has 298~300 frames
* (width, height) = (1920, 1080)

metadata.json

```
{
    {filename}: {
        "label": ("FAKE" or "REAL"),
        "split": "train"
    }
}
```