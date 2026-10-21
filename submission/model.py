"""Studio 2 submission: ConvNeXt-Tiny, five ordered waste classes.
Do not download pretrained weights here; evaluate.py loads model.pth.
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5

def get_model():
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])