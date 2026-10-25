"""Studio 2 - single timm ConvNeXt-Tiny pretrained in22k-ft-in1k architecture.
Evaluation never downloads weights; evaluate.py loads submission/model.pth.
"""
import timm
from torchvision import transforms

CLASSES = ['botol_plastik', 'gelas_plastik', 'kertas_kardus', 'saset_kemasan', 'styrofoam']
ARCH = 'convnext_tiny.fb_in22k_ft_in1k'

def get_model():
    return timm.create_model(ARCH, pretrained=False, num_classes=len(CLASSES))

def get_transform():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
