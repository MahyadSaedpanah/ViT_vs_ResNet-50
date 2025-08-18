import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from models.vit_model import get_vit_model
import os

def show_vit_attention(model, image: Image.Image, device):
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    attn_weights = []

    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            B, N, C = output.shape
            if hasattr(module, 'qkv'):
                qkv = module.qkv(input[0])  # (B, N, 3*dim)
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn.softmax(dim=-1)
                attn_weights.append(attn.detach().cpu())

    handle = model.blocks[-1].attn.register_forward_hook(forward_hook)

    with torch.no_grad():
        _ = model(img_tensor)

    handle.remove()

    if not attn_weights:
        print("⚠️ No attention weights captured.")
        return

    attn_map = attn_weights[0][0]  # (heads, tokens, tokens)
    attn_cls = attn_map[:, 0, 1:]  # cls token attention to patches
    attn_mean = attn_cls.mean(0).reshape(14, 14).numpy()
    attn_mean = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min())

    plt.imshow(image.resize((224, 224)))
    plt.imshow(attn_mean, cmap='jet', alpha=0.5, extent=(0, 224, 224, 0))
    plt.title("ViT Attention Map (Last Layer)")
    plt.axis('off')
    plt.savefig("vit_attention_output.png")
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vit_model()

    try:
        model.load_state_dict(torch.load("checkpoints/best_vit.pth", map_location=device))
        print("Loaded trained model weights from checkpoints/best_vit.pth")
    except:
        print("Failed to load trained weights. Using pretrained ViT instead.")

    if not os.path.exists("sample.jpg"):
        print("Downloading default sample image...")
        import urllib.request
        urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg", "sample.jpg")

    image = Image.open("sample.jpg").convert("RGB")
    show_vit_attention(model, image, device)
