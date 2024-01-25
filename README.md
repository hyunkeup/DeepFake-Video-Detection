# DeepFake Detection

## Original Dataset information

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

## How to start

### 1. Start the preprocessing

```
python SupervisedLearning/run_preprocessor.py

# For each divided folder. It will take at least 360 seconds 
```

### 2. Train

```
python SupervisedLearning/train_SimpleCNN.py, train/train_RestNet18.py, or ....
```

### 3. Test

```
python SupervisedLearning/test_SimpleCNN.py, test/test_RestNet18.py, or ....
```

## Model

### SimpleCNN

#### Dataset1:

``` 
Train: train_sample_images_with_mediapipe(train_sample_images_with_mediapipe)
Test: train_sample_images_with_mediapipe(train_sample_images_with_mediapipe)
```

#### Accuracy

Path:

```
```

### RestNet18

#### Dataset1:

```
Train: train_sample_images_with_mediapipe(train_sample_images_with_mediapipe)
Test: train_sample_images_with_mediapipe(train_sample_images_with_mediapipe)
```

#### Accuracy

Path: [ResNet18_model_20240121_001621.pth](test%2FResNet18_model_20240121_001621.pth)

```
# num_of_epochs: 100, batch_size: 32, 
# Accuracy: 100.0%
```