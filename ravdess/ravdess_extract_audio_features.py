import datetime
import glob
import os

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.signal import resample
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from tqdm import tqdm

#################### Hyperparameters ###################
BATCH_SIZE = 32
LEARNING_RATE = 0.04
N_EPOCH = 250
LR_STEPS = [40, 55, 65, 70, 200, 250]
########################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc


def audio_load(file_path):
    '''
    Loads and resample audio if not at 44100
    '''
    target_sr = 44100
    try:
        audio_data, sr = librosa.load(file_path, sr=None, mono=True)
        # check sampling rate is 44100Hz
        if sr != target_sr:
            # resample audio to 44100 (easier to work with to match with video frames)
            num_samples = len(audio_data)
            audio_data = resample(audio_data, int(num_samples * target_sr / sr))
        return audio_data, target_sr  # resampled audio data, target sampling rate
    except Exception as e:
        print(f"Error: {e}")


def get_ravdess_metadata(dir_path):
    # https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data
    """
        0: Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
        1: Vocal channel (01 = speech, 02 = song).
        2*: Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        3: Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        4: Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        5: Repetition (01 = 1st repetition, 02 = 2nd repetition).
        6: Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    :param dir_path:
    :return:
    """
    metadata = []
    for wav_file in glob.glob(os.path.join(dir_path, "*.wav")):
        directory_path = os.path.dirname(wav_file)
        filename = os.path.basename(wav_file)
        filename_without_extension = filename.split(".")[0]

        # emotion index is 2
        data = {
            "directory_path": directory_path,
            "filename": filename,
            "label": int(filename.split(".")[0].split("-")[2]) - 1,
            "wav_file_path": wav_file,
            "image_file_path": os.path.join(directory_path, filename_without_extension + ".png"),
            "np_file_path": os.path.join(directory_path, filename_without_extension + ".npy")
        }
        metadata.append(data)

    return metadata


def generate_datasets(dir_path, min_duration=3):
    print("=" * 50 + " Generating datasets " + "=" * 50)
    print(f"From {dir_path}")
    metadata = get_ravdess_metadata(dir_path)

    for data in tqdm(metadata):
        wav_file_path = data["wav_file_path"]
        image_file_path = data["image_file_path"]
        np_file_path = data["np_file_path"]

        # Load the audio file
        audio, sr = audio_load(wav_file_path)  # 3 seconds

        # Padding or Cutting by min_duration
        audio_length = min_duration * sr
        if audio.shape[0] < min_duration * sr:
            # Padding
            audio = np.pad(audio, (0, audio_length - len(audio)), 'constant')
        elif audio.shape[0] > audio_length:
            # Cutting
            audio = audio[:audio_length]

        audio_frames = librosa.util.frame(audio, frame_length=sr, hop_length=sr)
        audio_frames = np.transpose(audio_frames)
        assert audio_frames.shape[0] == min_duration, f"The audio frames should have {min_duration} seconds duration."

        # Extract audio features
        audio_features = [get_mfccs(y=audio_frame, sr=sr) for audio_frame in audio_frames]

        # Plotting MFCC features
        plt.clf()
        plt.axis("off")
        plt.imshow(np.vstack(audio_features), interpolation="nearest", origin="lower", aspect="auto")
        plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)

        # Resize for 244 x 244
        frame = cv2.imread(image_file_path)
        frame = cv2.resize(frame, (244, 244))
        cv2.imwrite(image_file_path, frame)
        np.save(np_file_path, np.array(frame))

    print("=" * 50 + " Datasets are generated. " + "=" * 50)


class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = torchvision.models.resnet18(num_classes=8)

    def forward(self, x):
        return self.model(x)


class RAVDESS(data.Dataset):
    def __init__(self, metadata, transform=None):
        self.data = metadata
        self.transform = transform

    def __getitem__(self, index):
        x = np.load(self.data[index]["np_file_path"])
        if self.transform is not None:
            image = Image.fromarray(x)
            x = self.transform(image)
        else:
            x = x.transpose(2, 0, 1)
        y = self.data[index]["label"]
        return x, y

    def __len__(self):
        return len(self.data)


def adjust_learning_rate(optimizer, epoch, learning_rate, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


def train(criterion, model, optimizer, train_loader):
    print("=" * 40 + " Hyperparameters " + "=" * 40)
    print(f"Batch size: {BATCH_SIZE}\nLearning rate: {LEARNING_RATE}\nNumber of epochs: {N_EPOCH}\ndevice: {device}")
    print("=" * 97)
    for epoch in range(N_EPOCH):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch, learning_rate=LEARNING_RATE, lr_steps=LR_STEPS)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{N_EPOCH}, lr: {lr}")

        model.train()
        running_loss = 0.0
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Loss: {running_loss / len(train_loader)}\n")
    model_file = f"./RAVDESS_bs_{BATCH_SIZE}_lr_{LEARNING_RATE}_ep_{N_EPOCH}_{datetime.datetime.now().strftime('%m-%d %H %M %S')}.pth"
    torch.save(model, model_file)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    pred_scores = []
    true_labels = []

    print("Evaluate:")
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

            true_labels.extend(y.cpu().numpy())
            pred_scores.extend(outputs.cpu().numpy())

    accuracy = correct / total
    pred_scores = torch.tensor(pred_scores)
    true_labels = torch.tensor(true_labels)
    _, pred_top5 = pred_scores.topk(5, 1)
    top1_accuracy = (pred_top5[:, 0] == true_labels).float().mean().item()
    top5_accuracy = (pred_top5 == true_labels.view(-1, 1)).float().sum(1).mean().item()

    y_scores = []
    for y_true, y_score in zip(true_labels.numpy(), pred_scores.numpy()):
        y_scores.append(y_score[y_true])
    auroc = roc_auc_score(true_labels.numpy(), np.array(y_scores), multi_class='ovr')

    return accuracy, top1_accuracy, top5_accuracy, auroc


def main():
    dir_path = "C:\\workspace\\deepfake-detection-challenge\\audio_resampled"
    metadata = get_ravdess_metadata(dir_path)
    # generate_datasets(dir_path)

    # Datasets
    image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_data, test_data = np.split(np.array(metadata), [int(len(metadata) * 0.8)])
    train_loader = torch.utils.data.DataLoader(RAVDESS(train_data, transform=image_transform),
                                               batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(RAVDESS(test_data, transform=image_transform),
                                              batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = ResNetModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    train(criterion=criterion, model=model, optimizer=optimizer, train_loader=train_loader)

    # Validation
    accuracy, top1_accuracy, top5_accuracy, auroc = evaluate(model=model, test_loader=test_loader)
    print(f'Accuracy: {accuracy}')
    print(f'Top-1 Accuracy: {top1_accuracy}')
    print(f'Top-5 Accuracy: {top5_accuracy}')
    print(f'AUROC: {auroc}')


if __name__ == "__main__":
    main()
