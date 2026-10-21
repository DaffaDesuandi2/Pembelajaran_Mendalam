"""Train ConvNeXt-Tiny on the SAME my_train/my_val split as ResNet baseline.

Usage (Windows VS Code):
  python train_convnext.py --train ./my_train --val ./my_val --batch-size 2
  python evaluate.py --submission ./submission --data ./my_val --batch-size 2

Training saves the best local Macro-F1 to submission/model.pth.
The top-level submission/ is overwritten when a better checkpoint is saved.
"""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from submission.model import get_model, get_transform

EXPECTED_CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
SAVE_PATH = Path("submission/model.pth")


def set_trainable(model, finetune):
    # Stage 1: classifier only. Stage 2: classifier + final ConvNeXt stage.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if finetune:
        for p in model.features[7].parameters():
            p.requires_grad = True


def train_epoch(model, loader, optimizer, criterion, scaler, device, accumulation, finetune):
    model.eval()  # frozen backbone remains in inference mode
    model.classifier.train()
    if finetune:
        model.features[7].train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            logits = model(x)
            loss = criterion(logits, y)
        # Correct normalization for final partial accumulation bucket.
        remaining = len(loader) % accumulation
        denom = remaining if (remaining and i >= len(loader) - remaining) else accumulation
        scaler.scale(loss / denom).backward()
        if (i + 1) % accumulation == 0 or i + 1 == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.item() * len(x)
    return loss_sum / len(loader.dataset)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    truth, pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            logits = model(x)
        pred.extend(logits.argmax(1).cpu().tolist())
        truth.extend(y.tolist())
    return accuracy_score(truth, pred), f1_score(truth, pred, labels=list(range(5)), average='macro', zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='./my_train')
    ap.add_argument('--val', default='./my_val')
    ap.add_argument('--head-epochs', type=int, default=4)
    ap.add_argument('--finetune-epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=2, help='Reduce to 1 if CUDA OOM')
    ap.add_argument('--accumulation', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} | GPU={torch.cuda.get_device_name(0) if device.type == "cuda" else "none"}')
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr = datasets.ImageFolder(a.train, transform=train_tf)
    va = datasets.ImageFolder(a.val, transform=get_transform())
    if tr.classes != EXPECTED_CLASSES or va.classes != EXPECTED_CLASSES:
        raise RuntimeError(f'Wrong class order: train={tr.classes} val={va.classes}')
    print(f'train={len(tr)} val={len(va)} classes={tr.classes}')
    tr_dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True,
                       num_workers=a.workers, pin_memory=device.type == 'cuda')
    va_dl = DataLoader(va, batch_size=a.batch_size, shuffle=False,
                       num_workers=a.workers, pin_memory=device.type == 'cuda')

    # Download ImageNet pretrained weights ONLY during training, not during evaluation.
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[2].in_features, 5)
    )
    model.to(device)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best_f1 = -1.0
    for stage, n_epochs in [('head', a.head_epochs), ('finetune', a.finetune_epochs)]:
        if n_epochs <= 0:
            continue
        finetune = stage == 'finetune'
        set_trainable(model, finetune)
        # The last ConvNeXt stage learns slower than the new classification head.
        groups = [{'params': model.classifier.parameters(), 'lr': 3e-4}]
        if finetune:
            groups.append({'params': model.features[7].parameters(), 'lr': 1e-5})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        for ep in range(1, n_epochs + 1):
            loss = train_epoch(model, tr_dl, opt, criterion, scaler, device, a.accumulation, finetune)
            acc, f1 = validate(model, va_dl, device)
            sched.step()
            print(f'{stage} {ep:02d}/{n_epochs:02d} | train_loss={loss:.4f} | val_acc={acc:.4f} | val_macro_f1={f1:.4f}', flush=True)
            if f1 > best_f1:
                best_f1 = f1
                # Float16 checkpoint cuts ConvNeXt-Tiny size to ~57 MB for normal GitHub uploads.
                # evaluate.py loads it into float32 model parameters without changes.
                compact_state = {
                    k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                    for k, v in model.state_dict().items()
                }
                torch.save(compact_state, SAVE_PATH)
                print(f'  saved best -> {SAVE_PATH}', flush=True)
    print(f'BEST local Macro-F1={best_f1:.4f} | checkpoint={SAVE_PATH}')


if __name__ == '__main__':
    main()
