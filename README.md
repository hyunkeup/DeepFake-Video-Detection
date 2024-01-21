# CSCI566_Final_Project

DeepFake Detector

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

## Data generalization

* 224x224: This size was widely used in early CNN models like AlexNet.
* 299x299: Some models, such as Inception, use this size.
* 224x224, 299x299, 331x331, 448x448: Some models support multiple sizes, and these are commonly used options.
* Custom Sizes: Depending on the project, custom sizes may be chosen. Some models are designed to handle a variety of
  input sizes.
* Generally, the choice of image size is influenced by the model architecture, characteristics of the training data,
  computational resources, and specific project requirements. When using transfer learning, it's often recommended to
  follow the expected input size of the pre-trained model. The image size should always be adjusted to suit the
  particular context and needs.

## How to start

### 1. Start the preprocessing

```
python preprocessing/run_preprocessor.py

# For each divided folder. It will take at least 360 seconds 
```

### 2. Train

```
python train/train_SimpleCNN.py, train/train_RestNet18.py, or ....
```

### 3. Test

```
python test/test_SimpleCNN.py, test/test_RestNet18.py, or ....
```

## Model

### SimpleCNN

#### Dataset1:

``` 
Train: train_sample_videos(train_sample_images_fullscreen)
Test: train_sample_videos(train_sample_images_fullscreen)
```
#### Accuracy
```
# num_of_epochs: 100, batch_size: 32
# Accuracy: 91.8%
# Path: [SimplCNN_model_20240120_154945.pth](test%2FSimplCNN_model_20240120_154945.pth)

# num_of_epochs: 100, batch_size: 32
# Accuracy: 95.2%
# Path: [SimpleCNN_model_20240120_181615.pth](test%2FSimpleCNN_model_20240120_181615.pth)
```

### RestNet18

#### Dataset1:

```
Train: train_sample_videos(train_sample_images_fullscreen)
Test: train_sample_videos(train_sample_images_fullscreen)
```
#### Accuracy
```
# num_of_epochs: 100, batch_size: 32
# Accuracy: 84.2%
# Model: [ResNet18_model_20240120_162732.pth](test%2FResNet18_model_20240120_162732.pth)

# num_of_epoch2: 200, batch_size: 32
# Accuracy: 88.0%
# Model: [ResNet18_model_20240120_164851.pth](test%2FResNet18_model_20240120_164851.pth)
```