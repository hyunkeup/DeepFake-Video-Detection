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

### 4. Summary

#### SimpleCNN

```
Accuracy: 91.8%
```

#### RestNet18

```
Accuracy: 84.2%
```