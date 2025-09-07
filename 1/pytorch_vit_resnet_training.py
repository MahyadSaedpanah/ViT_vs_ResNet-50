
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViT-Base-16 vs ResNet-50 on CIFAR-100 (PyTorch)
- Transfer learning (freeze) and full fine-tuning
- AdamW for ViT, SGD/AdamW for ResNet
- EarlyStopping on val acc
- LR scheduler (ReduceLROnPlateau)
- Metrics: Top-1 accuracy on train/val/test
- Timing per epoch and inference per sample
- Params count and FLOPs (thop)
- Robustness eval (Gaussian noise, random occlusion)
- Interpretability: ViT Attention Rollout (hook on Attention.attn_drop)
- Plots for loss/acc
"""

import os, time, argparse, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
from timm.models.vision_transformer import VisionTransformer
import matplotlib.pyplot as plt

try:
    from thop import profile
    THOP_AVAILABLE = True
except Exception:
    THOP_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item()

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0; self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n; self.cnt += n
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)

class EarlyStopping:
    def __init__(self, patience=7, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best = -float('inf')
        self.early_stop = False
        self.verbose = verbose
    def step(self, metric):
        if metric > self.best + 1e-8:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# -----------------------------
# Data
# -----------------------------

def get_transforms(img_size=224, is_train=True):
    if is_train:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                 std=(0.2673, 0.2564, 0.2762)),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                 std=(0.2673, 0.2564, 0.2762)),
        ])
    return tfm

def get_dataloaders(data_dir, img_size=224, batch_size=128, num_workers=4, val_split=0.1):
    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=get_transforms(img_size, True))
    test_set  = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=get_transforms(img_size, False))

    # Build a val split from train
    n_train = len(train_set)
    n_val = int(n_train * val_split)
    n_train = n_train - n_val
    train_subset, val_subset = torch.utils.data.random_split(train_set, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# -----------------------------
# Models
# -----------------------------

def build_model(model_name: str, num_classes=100, pretrained=True):
    if model_name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
        assert isinstance(model, VisionTransformer)
        return model
    elif model_name == "resnet":
        # Torchvision ResNet-50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # Replace FC
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError("model_name must be 'vit' or 'resnet'")

def freeze_backbone(model, model_name):
    if model_name == "vit":
        # Freeze all except classifier head
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:  # resnet
        for name, p in model.named_parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compute_flops(model, img_size=224):
    if not THOP_AVAILABLE:
        return None
    dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = macs * 2  # rough convert MACs->FLOPs
    return int(flops)

# -----------------------------
# Train / Eval
# -----------------------------

def epoch_pass(model, loader, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    start = time.time()

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)

        acc = accuracy(logits, labels)
        loss_meter.update(loss.item(), imgs.size(0))
        acc_meter.update(acc, imgs.size(0))

    elapsed = time.time() - start
    return loss_meter.avg, acc_meter.avg, elapsed

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        acc = accuracy(logits, labels)
        loss_meter.update(loss.item(), imgs.size(0))
        acc_meter.update(acc, imgs.size(0))
    return loss_meter.avg, acc_meter.avg

def plot_curves(history, out_dir):
    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(out_dir, 'acc_curve.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150)
    plt.close()

# -----------------------------
# Robustness
# -----------------------------

@torch.no_grad()
def eval_with_noise(model, loader, noise_std=0.1):
    model.eval()
    acc_meter = AverageMeter()
    for imgs, labels in loader:
        imgs = imgs + noise_std * torch.randn_like(imgs)
        imgs = torch.clamp(imgs, -5.0, 5.0)  # after normalization, keep bounded
        imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
        logits = model(imgs)
        acc = accuracy(logits, labels)
        acc_meter.update(acc, imgs.size(0))
    return acc_meter.avg

@torch.no_grad()
def eval_with_occlusion(model, loader, erase_size=32):
    model.eval()
    acc_meter = AverageMeter()
    for imgs, labels in loader:
        b, c, h, w = imgs.size()
        # Random square erase
        top = torch.randint(0, h - erase_size + 1, (b,))
        left = torch.randint(0, w - erase_size + 1, (b,))
        for i in range(b):
            imgs[i, :, top[i]:top[i]+erase_size, left[i]:left[i]+erase_size] = 0.0
        imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
        logits = model(imgs)
        acc = accuracy(logits, labels)
        acc_meter.update(acc, imgs.size(0))
    return acc_meter.avg

# -----------------------------
# ViT Attention Rollout
# -----------------------------

class VitAttentionHook:
    """Captures softmaxed attention tensors from timm ViT blocks via attn_drop pre-dropout input."""
    def __init__(self, model: VisionTransformer):
        self.handles = []
        self.attn_maps = []  # list of tensors [B, Heads, N, N]
        for blk in model.blocks:
            handle = blk.attn.attn_drop.register_forward_hook(self._hook)
            self.handles.append(handle)

    def _hook(self, module, inp, out):
        # inp[0] is attention AFTER softmax (pre-dropout): shape [B, Heads, N, N]
        self.attn_maps.append(inp[0].detach().cpu())

    def clear(self):
        self.attn_maps.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

def attention_rollout(attn_list, discard_ratio=0.0):
    # attn_list: list of [B, H, N, N], accumulate across layers
    # Implementation adapted from common rollout approach
    result = None
    for attn in attn_list:
        # Average over heads
        attn_mean = attn.mean(dim=1)  # [B, N, N]
        if discard_ratio > 0:
            flat = attn_mean.view(attn_mean.size(0), -1)
            num = int(flat.size(1) * discard_ratio)
            if num > 0:
                idx = flat.topk(num, dim=1, largest=False).indices
                flat.scatter_(1, idx, 0.0)
                attn_mean = flat.view_as(attn_mean)
        # Add identity
        I = torch.eye(attn_mean.size(-1)).unsqueeze(0).expand_as(attn_mean)
        attn_aug = attn_mean + I
        # Normalize
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        result = attn_aug if result is None else torch.bmm(result, attn_aug)
    return result  # [B, N, N]

@torch.no_grad()
def visualize_vit_attention(model, img_tensor, out_path="vit_attention.png", discard_ratio=0.0):
    model.eval()
    if not isinstance(model, VisionTransformer):
        print("Attention visualization only for ViT models.")
        return
    hook = VitAttentionHook(model)
    _ = model(img_tensor.to(DEVICE))  # forward to collect attentions
    attn_result = attention_rollout(hook.attn_maps, discard_ratio=discard_ratio)  # [B, N, N]
    hook.remove()
    # Token 0 is class token; map from CLS to patch tokens
    cls_attn = attn_result[:, 0, 1:]  # [B, N_patches]
    B = cls_attn.size(0)
    num_patches = cls_attn.size(1)
    grid_size = int(math.sqrt(num_patches))
    attn_maps = cls_attn.view(B, 1, grid_size, grid_size)  # [B,1,H,W]
    # Upsample to image size (assume square)
    attn_maps = F.interpolate(attn_maps, size=(img_tensor.size(-2), img_tensor.size(-1)), mode='bilinear', align_corners=False)
    # Save first image heatmap
    attn_map = attn_maps[0,0].cpu().numpy()
    fig = plt.figure()
    plt.imshow(attn_map, cmap='jet', alpha=0.7)
    plt.axis('off')
    plt.title('ViT Attention Rollout')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Inference timing
# -----------------------------

@torch.no_grad()
def measure_inference_time(model, img_size=224, repeats=100):
    model.eval()
    dummy = torch.randn(1,3,img_size,img_size).to(DEVICE)
    # warmup
    for _ in range(10):
        _ = model(dummy)
    torch.cuda.synchronize() if DEVICE.type=='cuda' else None
    start = time.time()
    for _ in range(repeats):
        _ = model(dummy)
    torch.cuda.synchronize() if DEVICE.type=='cuda' else None
    elapsed = time.time() - start
    return elapsed / repeats

# -----------------------------
# Main training routine
# -----------------------------

def train_and_eval(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.img_size, args.batch_size, args.num_workers, args.val_split)

    model = build_model(args.model, num_classes=100, pretrained=True).to(DEVICE)

    # Freeze stage (train head only)
    if args.freeze_stage:
        freeze_backbone(model, args.model)

    total_params, trainable_params = count_params(model)
    print(f"Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")

    if args.model == "vit":
        # Default to AdamW with smaller LR for ViT
        lr = args.lr if args.lr is not None else (1e-3 if args.freeze_stage else 5e-5)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.weight_decay)
    else:
        # ResNet: SGD with momentum (or AdamW if --adamw)
        lr = args.lr if args.lr is not None else (1e-3 if args.freeze_stage else 1e-3)
        if args.adamw:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    early = EarlyStopping(patience=args.es_patience, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = -1.0
    best_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    print("=== Stage 1: Head-only training ===" if args.freeze_stage else "=== Single-stage training ===")
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc, tr_time = epoch_pass(model, train_loader, optimizer, scaler)
        va_loss, va_acc = evaluate(model, val_loader)
        scheduler.step(va_acc)

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'  ].append(va_loss); history['val_acc'  ].append(va_acc)

        print(f"Epoch {epoch:03d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} | epoch_time={tr_time:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"  >> Saved best to {best_path} (val_acc={best_val_acc:.4f})")

        early.step(va_acc)
        if early.early_stop:
            print("Early stopping triggered.")
            break

    # Optional Stage 2: full fine-tuning
    if args.finetune_stage:
        print("=== Stage 2: Full fine-tuning ===")
        # Load best from stage 1
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        unfreeze_all(model)
        # Recreate optimizer with smaller LR
        if args.model == "vit":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr or 5e-5, weight_decay=args.weight_decay)
        else:
            if args.adamw:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr or 1e-4, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.ft_lr or 1e-3, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        early = EarlyStopping(patience=args.es_patience, verbose=True)

        for epoch in range(1, args.ft_epochs+1):
            tr_loss, tr_acc, tr_time = epoch_pass(model, train_loader, optimizer, scaler)
            va_loss, va_acc = evaluate(model, val_loader)
            scheduler.step(va_acc)

            history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
            history['val_loss'  ].append(va_loss); history['val_acc'  ].append(va_acc)

            print(f"[FT] Epoch {epoch:03d}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} | epoch_time={tr_time:.1f}s")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), best_path)
                print(f"  >> Saved best to {best_path} (val_acc={best_val_acc:.4f})")

            early.step(va_acc)
            if early.early_stop:
                print("Early stopping (fine-tune) triggered.")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Curves
    plot_curves(history, args.out_dir)

    # FLOPs & inference time
    flops = compute_flops(model, args.img_size)
    if flops is not None:
        with open(os.path.join(args.out_dir, f"{args.model}_flops_params.txt"), "w") as f:
            f.write(f"FLOPs (approx): {flops}\n")
            total_params, trainable_params = count_params(model)
            f.write(f"Total params: {total_params}\nTrainable params: {trainable_params}\n")
    inf_t = measure_inference_time(model, args.img_size, repeats=50)
    with open(os.path.join(args.out_dir, f"{args.model}_timing.txt"), "w") as f:
        f.write(f"Inference time per sample (s): {inf_t}\n")

    # Robustness
    noise_acc = eval_with_noise(model, test_loader, noise_std=0.1)
    occ_acc   = eval_with_occlusion(model, test_loader, erase_size=32)
    with open(os.path.join(args.out_dir, f"{args.model}_robustness.txt"), "w") as f:
        f.write(f"Gaussian noise std=0.1 acc: {noise_acc}\n")
        f.write(f"Occlusion 32x32 acc: {occ_acc}\n")

    # Attention maps for ViT
    if args.model == "vit":
        # take one batch from test
        imgs, _ = next(iter(test_loader))
        img0 = imgs[0:1]
        visualize_vit_attention(model, img0.to(DEVICE), out_path=os.path.join(args.out_dir, "vit_attention.png"))

    print("Done. Outputs saved to:", args.out_dir)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="vit", choices=["vit","resnet"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--freeze_stage", action="store_true", help="Stage 1: train head only with frozen backbone")
    p.add_argument("--finetune_stage", action="store_true", help="Stage 2: full fine-tune")
    p.add_argument("--ft_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=None, help="LR for initial stage; default depends on model & stage")
    p.add_argument("--ft_lr", type=float, default=None, help="LR for fine-tune stage")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--adamw", action="store_true", help="Use AdamW for ResNet too")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--es_patience", type=int, default=5)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_eval(args)
