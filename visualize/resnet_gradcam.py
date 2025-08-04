import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models.resnet_model import get_resnet_model
from torchvision.models.feature_extraction import create_feature_extractor


def apply_gradcam(model, image: Image.Image, device):
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Register hook on the last conv layer
    target_layer = "layer4"
    return_nodes = {target_layer: "feat"}
    extractor = create_feature_extractor(model, return_nodes=return_nodes)

    activations = {}
    def hook_fn(module, input, output):
        activations['value'] = output

    handle = extractor[target_layer].register_forward_hook(hook_fn)

    # Forward pass
    output = extractor(input_tensor)
    logits = model(input_tensor)
    class_idx = torch.argmax(logits, dim=1).item()

    # Backward pass
    model.zero_grad()
    logits[0, class_idx].backward()
    gradients = extractor[target_layer].weight.grad  # (out_channels, in_channels, k, k)

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activation = activations['value'].squeeze(0)

    for i in range(activation.shape[0]):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = activation.mean(dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((224, 224), Image.Resampling.BILINEAR)

    plt.imshow(image.resize((224, 224)))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM - ResNet")
    plt.axis('off')
    plt.show()

    handle.remove()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model()
    model.load_state_dict(torch.load("best_resnet.pth", map_location=device))
    image = Image.open("sample.jpg").convert("RGB")
    apply_gradcam(model, image, device)
