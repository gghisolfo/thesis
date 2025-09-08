import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn

from torchvision.models.segmentation.deeplabv3 import DeepLabHead


#num_classes: put 8 or the number of categories you have
#Works on input size ≥ 224×224.

# def get_deeplabv3plus_model(num_classes, pretrained=True):
#     model = deeplabv3_resnet50(pretrained=pretrained)
#     model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
#     return model


def get_deeplabv3plus_model(in_channels, num_classes):
    model = deeplabv3_resnet50(weights=None)  # No pretraining unless you're handling weights correctly
    model.classifier = DeepLabHead(2048, num_classes)  # Fully replace classifier head
    return model

