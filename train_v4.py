"""Train SINGLE timm ConvNeXt-Tiny ImageNet-21K->1K. Saves to experiments/v4/submission.
Example: python train_v4.py --train my_train --val my_val --batch-size 1 --accumulation 16
Never trains on validation images. Keep official evaluate.py unchanged.
"""
import argparse
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score
import timm
from submission_v4_model import ARCH, get_transform

CLASSES = ['botol_plastik','gelas_plastik','kertas_kardus','saset_kemasan','styrofoam']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, scaler, criterion, device, accumulation, head_only):
    model.train()
    if head_only:
        # Frozen backbone uses eval mode; only the classifier learns.
        model.eval(); model.get_classifier().train()
    optimizer.zero_grad(set_to_none=True)
    running=0.
    for i,(x,y) in enumerate(loader):
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.autocast('cuda',enabled=device.type=='cuda'):
            loss=criterion(model(x),y)
        remainder=len(loader)%accumulation
        denom=remainder if remainder and i>=len(loader)-remainder else accumulation
        scaler.scale(loss/denom).backward()
        if (i+1)%accumulation==0 or i+1==len(loader):
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        running+=loss.item()*len(y)
        if (i+1)%200==0: print(f'  batches {i+1}/{len(loader)}', flush=True)
    return running/len(loader.dataset)

@torch.inference_mode()
def val_score(model,loader,device):
    model.eval(); truth=[]; pred=[]
    for x,y in loader:
        with torch.autocast('cuda',enabled=device.type=='cuda'):
            logits=model(x.to(device,non_blocking=True))
        pred+=logits.argmax(1).cpu().tolist(); truth+=y.tolist()
    return accuracy_score(truth,pred), f1_score(truth,pred,labels=list(range(5)),average='macro',zero_division=0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='./my_train'); p.add_argument('--val',default='./my_val')
    p.add_argument('--out',default='experiments/v4/submission')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--head-epochs',type=int,default=2)
    p.add_argument('--finetune-epochs',type=int,default=18)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--backbone-lr',type=float,default=2e-5)
    p.add_argument('--full-finetune',action='store_true',help='Train all backbone; may OOM on 4GB GPU')
    a=p.parse_args(); seed_all(a.seed)
    if not torch.cuda.is_available(): print('WARNING: CUDA unavailable; training on CPU will be slow.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'GPU={torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"}',flush=True)
    tr_tf=transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.75,1.0)),transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),transforms.ColorJitter(0.15,0.15,0.15),
        transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train=datasets.ImageFolder(a.train,transform=tr_tf)
    val=datasets.ImageFolder(a.val,transform=get_transform())
    if train.classes!=CLASSES or val.classes!=CLASSES:
        raise ValueError(f'Unexpected class order: {train.classes}, {val.classes}')
    tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=='cuda')
    va=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda')
    print('Downloading pretrained weights if not cached...',flush=True)
    model=timm.create_model(ARCH,pretrained=True,num_classes=5).to(device)
    # ConvNeXt timm models expose backbone in stages: stem/stages and head classifier.
    head_params=list(model.get_classifier().parameters())
    if not head_params: raise RuntimeError('timm classifier parameters not found')
    head_ids={id(x) for x in head_params}
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    # Keep a matching architecture definition alongside weights for evaluate.py.
    src=Path(__file__).parent/'submission_v4_model.py'
    import shutil
    shutil.copyfile(src,out/'model.py')
    best=-1.
    for stage,num_epochs in [('head',a.head_epochs),('finetune',a.finetune_epochs)]:
        if num_epochs<=0: continue
        for p0 in model.parameters(): p0.requires_grad=False
        for p0 in head_params: p0.requires_grad=True
        if stage=='finetune':
            if a.full_finetune:
                for p0 in model.parameters(): p0.requires_grad=True
            else:
                for p0 in model.stages[-1].parameters(): p0.requires_grad=True
        other=[p0 for p0 in model.parameters() if p0.requires_grad and id(p0) not in head_ids]
        groups=[{'params':head_params,'lr':a.head_lr}]
        if other: groups.append({'params':other,'lr':a.backbone_lr})
        optimizer=torch.optim.AdamW(groups,weight_decay=0.05)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs)
        for ep in range(1,num_epochs+1):
            loss=train_epoch(model,tr,optimizer,scaler,criterion,device,a.accumulation,stage=='head')
            acc,f1=val_score(model,va,device)
            scheduler.step()
            print(f'{stage} {ep:02}/{num_epochs:02} loss={loss:.4f} val_acc={acc:.4f} val_macro_f1={f1:.4f}',flush=True)
            if f1>best:
                best=f1
                compact={k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k,v in model.state_dict().items()}
                torch.save(compact,out/'model.pth')
                print(f'  NEW BEST {best:.4f} -> {out / "model.pth"}',flush=True)
    print(f'DONE best_macro_f1={best:.4f}; evaluate: python evaluate.py --submission {out} --data {a.val} --batch-size 2')

if __name__=='__main__':main()
