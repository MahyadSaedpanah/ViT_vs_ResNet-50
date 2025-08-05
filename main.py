import yaml
import torch
import logging
import os
from datetime import datetime
from data.prepare_data import get_cifar100_dataloaders
from models.vit_model import get_vit_model
from models.resnet_model import get_resnet_model
from train.train_vit import train as train_vit
from train.train_resnet import train as train_resnet
from evaluate.evaluate import evaluate_model
from utils.helpers import set_seed

# Make logger global so train scripts can reuse it
log = logging.getLogger()

if __name__ == "__main__":
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/train_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()
    log.info(f"Log file started: {log_file}")

    # Load config
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['training']['seed'])
    device = torch.device(cfg['hardware']['device'] if torch.cuda.is_available() else 'cpu')

    subset_ratio = cfg['training'].get('subset_ratio', 1.0)

    # Data
    log.info("Loading CIFAR-100 dataset...")
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(
        batch_size=cfg['training']['batch_size'],
        val_split=cfg['training']['val_split'],
        subset_ratio=subset_ratio,
        num_workers=cfg['hardware']['num_workers']
    )

    # --- ViT ---
    log.info("\nTraining ViT...")
    vit_model = get_vit_model(num_classes=cfg['dataset']['num_classes'])
    train_vit(
        model=vit_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg['training']['epochs'],
        lr=float(cfg['optimizer']['vit']['lr']),
        weight_decay=float(cfg['optimizer']['vit']['weight_decay']),
        freeze=cfg['optimizer']['vit']['freeze'],
        logger=log
    )

    acc, _, _ = evaluate_model(vit_model, test_loader, device)
    log.info(f"[ViT] Test Accuracy: {acc:.2f}%")

    # --- ResNet ---
    log.info("\nTraining ResNet...")
    resnet_model = get_resnet_model(num_classes=cfg['dataset']['num_classes'])
    train_resnet(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg['training']['epochs'],
        lr=float(cfg['optimizer']['resnet']['lr']),
        optimizer_type=cfg['optimizer']['resnet']['type'],
        weight_decay=float(cfg['optimizer']['resnet']['weight_decay']),
        logger=log
    )

    acc, _, _ = evaluate_model(resnet_model, test_loader, device)
    log.info(f"[ResNet] Test Accuracy: {acc:.2f}%")
