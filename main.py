import yaml
import torch
from data.prepare_data import get_cifar100_dataloaders
from models.vit_model import get_vit_model
from models.resnet_model import get_resnet_model
from train.train_vit import train as train_vit
from train.train_resnet import train as train_resnet
from evaluate import evaluate_model
from utils.helpers import set_seed

if __name__ == "__main__":
    # Load config
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['training']['seed'])
    device = torch.device(cfg['hardware']['device'] if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(
        batch_size=cfg['training']['batch_size'],
        val_split=cfg['training']['val_split'],
        num_workers=cfg['hardware']['num_workers']
    )

    # --- ViT ---
    print("\nTraining ViT...")
    vit_model = get_vit_model(num_classes=cfg['dataset']['num_classes'])
    train_vit(
        model=vit_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg['training']['epochs'],
        lr=cfg['optimizer']['vit']['lr'],
        weight_decay=cfg['optimizer']['vit']['weight_decay'],
        freeze=cfg['optimizer']['vit']['freeze']
    )

    acc, _, _ = evaluate_model(vit_model, test_loader, device)
    print(f"[ViT] Test Accuracy: {acc:.2f}%")

    # --- ResNet ---
    print("\nTraining ResNet...")
    resnet_model = get_resnet_model(num_classes=cfg['dataset']['num_classes'])
    train_resnet(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg['training']['epochs'],
        lr=cfg['optimizer']['resnet']['lr'],
        optimizer_type=cfg['optimizer']['resnet']['type'],
        weight_decay=cfg['optimizer']['resnet']['weight_decay']
    )

    acc, _, _ = evaluate_model(resnet_model, test_loader, device)
    print(f"[ResNet] Test Accuracy: {acc:.2f}%")
