import torch
import torch.nn as nn
from marlin_pytorch import Marlin
from torchvision.models import resnet18

from model.transformer_timm import AttentionBlock, Attention


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True))


class VideoMarlin(nn.Module):
    def __init__(self, input_channels=10, num_classes=2, marlin_model="marlin_vit_small_ytf"):
        super(VideoMarlin, self).__init__()
        self.marlin = Marlin.from_online(marlin_model)

        self.conv1d_0 = conv1d_block(input_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x):
        x = self.marlin.extract_features(x)
        return x

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='valid'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


class AudioResNet18(nn.Module):
    def __init__(self, input_channels=10, num_classes=2):
        super(AudioResNet18, self).__init__()
        self.input_channels = input_channels

        self.resnet18 = nn.Sequential(*list(resnet18().children())[:-1])
        self.resnet18_fc = nn.Linear(512, 1000)

        self.conv1d_0 = conv1d_block_audio(self.input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward_features(self, x):
        x = self.resnet18(x)
        x = x.view(x.shape[0], -1)
        x = self.resnet18_fc(x)
        x = x.reshape((x.shape[0], self.input_channels, int(x.shape[1] / self.input_channels)))
        return x

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


class FusionMultiModalCNN(nn.Module):
    def __init__(self, num_classes=2, fusion="it", e_dim=128, input_dim_video=128, input_dim_audio=128, num_heads=1,
                 marlin_model="marlin_vit_small_ytf"):
        """
        reference: https://github.com/katerynaCh/multimodal-emotion-recognition/tree/main

        :param num_classes: Final classes. e.g) FAKE or REAL in DeepFake
        :param fusion: "lt", "it", or "ia". You can see the architecture in the above reference.
        :param e_dim: It is out dim before num_classes
        :param input_dim_video:
        :param input_dim_audio:
        """
        super(FusionMultiModalCNN, self).__init__()
        self.num_classes = num_classes
        self.fusion = fusion
        self.e_dim = e_dim
        self.input_dim_video = input_dim_video
        self.input_dim_audio = input_dim_audio

        # MARLIN
        self.video_model = VideoMarlin(input_channels=1568, marlin_model=marlin_model)
        # ResNet18?
        self.audio_model = AudioResNet18(input_channels=10)

        if fusion == "lt":
            self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
                                     num_heads=num_heads)
            self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
                                     num_heads=num_heads)
        if fusion == "it":
            input_dim_video = input_dim_video // 2
            self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                                     num_heads=num_heads)
            self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
                                     num_heads=num_heads)
        if fusion == "ia":
            input_dim_video = input_dim_video // 2
            self.av = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                                num_heads=num_heads)
            self.va = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
                                num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(e_dim * 2, num_classes)
        )

    def _forward_lt(self, x_audio, x_video):
        pass

    def _forward_it(self, x_audio, x_video):
        # Extract features
        x_audio = self.audio_model.forward_features(x_audio)  # (32, 3, 244, 244) -> (32, 10, 1000)
        x_video = self.video_model.forward_features(x_video)  # (32, 3, 16, 244, 244) - > (32, 1568, 384)

        # Stage 1
        x_audio = self.audio_model.forward_stage1(x_audio)  # (32, 10, 1000) -> (32, 128, 94)
        x_video = self.video_model.forward_stage1(x_video)  # (32, 1568, 384) -> (32, 1568, 384)

        # Transformer
        proj_x_v = x_video.permute(0, 2, 1)
        proj_x_a = x_audio.permute(0, 2, 1)

        h_va = self.va(proj_x_a, proj_x_v)
        h_av = self.av(proj_x_v, proj_x_a)

        h_va = h_va.permute(0, 2, 1)
        h_av = h_av.permute(0, 2, 1)

        x_video = h_va + x_video
        x_audio = h_av + x_audio

        # Stage 2
        x_video = self.video_model.forward_stage2(x_video)
        x_audio = self.audio_model.forward_stage2(x_audio)

        video_pooled = x_video.mean([-1])
        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension

        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x = self.classifier(x)
        return x

    def _forward_ia(self, x_audio, x_video):
        pass

    def forward(self, x_audio, x_video):
        if self.fusion == "lt":
            return self._forward_lt(x_audio=x_audio, x_video=x_video)

        if self.fusion == "it":
            return self._forward_it(x_audio=x_audio, x_video=x_video)

        if self.fusion == "ia":
            return self._forward_ia(x_audio=x_audio, x_video=x_video)


def generate_model(device, num_classes=2, fusion="it", num_heads=1, marlin_model="marlin_vit_small_ytf"):
    model = FusionMultiModalCNN(num_classes=num_classes, fusion=fusion, num_heads=num_heads, marlin_model=marlin_model)

    if device != 'cpu':
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                   p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

    return model, model.parameters()
