import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models.resnet_model import get_resnet_model
import torch.nn.functional as F
import os

def apply_gradcam(model, image: Image.Image, device):
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def save_activation(module, input, output):
        activations.append(output.detach())

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.layer4[2].conv3
    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()
    model.zero_grad()
    output[0, class_idx].backward()

    handle_forward.remove()
    handle_backward.remove()

    if not gradients or not activations:
        print("No activations or gradients captured.")
        return

    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = F.relu(cam)

    if cam.max() == 0:
        print("CAM is zero. No informative gradients.")
        return

    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.cpu().numpy()
    cam = np.uint8(255 * cam)
    cam = Image.fromarray(cam).resize((224, 224), Image.Resampling.BILINEAR)

    plt.imshow(image.resize((224, 224)))
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM - ResNet")
    plt.axis('off')
    plt.savefig("gradcam_output.png")
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet_model()

    try:
        model.load_state_dict(torch.load("checkpoints/best_resnet.pth", map_location=device))
        print("Loaded trained model weights from checkpoints/best_resnet.pth")
    except:
        print("Failed to load trained weights. Using pretrained ResNet instead.")

    if not os.path.exists("sample.jpg"):
        print("Downloading default sample image...")
        import urllib.request
        urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg", "sample.jpg")

    image = Image.open("sample.jpg").convert("RGB")
    apply_gradcam(model, image, device)
