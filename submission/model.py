import torch

model = None  # global variable untuk menyimpan model


def get_model():
    """Bangun arsitektur saja — JANGAN mengunduh bobot atau memuat model.pth di sini."""
    ...
    return model  # output: logits [batch, 5]


def get_transform():
    """Transform validasi: PIL.Image (RGB) -> Tensor [3, H, W]."""
    ...


# Urutan kelas (alfabetis, sama dengan torchvision ImageFolder):
# 0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam

# Simpan bobot dengan:
torch.save(model.state_dict(), "submission/model.pth")
