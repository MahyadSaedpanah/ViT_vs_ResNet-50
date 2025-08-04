import torch
import time
import torch.nn as nn
from ptflops import get_model_complexity_info

def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    elapsed_time = time.time() - start_time
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    avg_time_per_sample = elapsed_time / total

    return accuracy, avg_loss, avg_time_per_sample

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops(model, input_res=(3, 224, 224)):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
    return macs, params

if __name__ == "__main__":
    from models.vit_model import get_vit_model
    from data.prepare_data import get_cifar100_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_cifar100_dataloaders()
    model = get_vit_model()
    model.load_state_dict(torch.load("best_vit.pth", map_location=device))

    acc, loss, time_per_sample = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Avg Loss: {loss:.4f}")
    print(f"Avg Inference Time per Sample: {time_per_sample:.6f} sec")

    print("Parameters:", count_parameters(model))
    macs, params = compute_flops(model)
    print("FLOPs:", macs)
    print("Model Params:", params)
