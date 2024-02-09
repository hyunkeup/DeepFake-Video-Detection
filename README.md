# DeepFake Detection

## Install
```
# Pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Others
pip install numpy opencv-python moviepy librosa facenet_pytorch
```

## Property
1. {env} is from python command line.
   * ```
        python3 extract_audios.py dev     
     ```
2. See appsettings_{env}.json
3. Call a function, get_property({key}) in Property class

## Original Dataset information from Dfdc
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