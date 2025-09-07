import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.prepare_data import get_cifar100_dataloaders
from models.vit_model import get_vit_model
import os

def train(model, train_loader, val_loader, device, epochs=30, lr=5e-5, weight_decay=0.01, freeze=False, logger=None):
    model.to(device)

    if freeze:
        for name, param in model.named_parameters():
            if "blocks" in name or "patch_embed" in name:
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = os.path.join("checkpoints", "best_vit.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0 
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            num_samples  += labels.size(0)


            # running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / num_samples

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device)

        # print(f"Epoch {epoch+1}: Train Loss={running_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        # message = f"Epoch {epoch+1}: Train Loss={running_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%"
        message = (
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Train Acc={train_acc:.2f}%, "
            f"Val Acc={val_acc:.2f}%"
        )
        if logger:
            logger.info(message)
        else:
            print(message)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # torch.save(model.state_dict(), "best_vit.pth")
            torch.save(model.state_dict(), best_model_path)
            if logger:
                logger.info(f"Best model saved to {best_model_path}")
            else:
                print(f"Best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if logger:
                    logger.info("Early stopping triggered.")
                else:
                    print("Early stopping triggered.")
                break

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_cifar100_dataloaders()
    model = get_vit_model(num_classes=100)
    train(model, train_loader, val_loader, device, freeze=False)
