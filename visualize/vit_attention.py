import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from models.vit_model import get_vit_model

def show_attention_map(model: VisionTransformer, image: Image.Image, device):
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.forward_features(img_tensor)  # (B, N, C)
        attn_maps = model.blocks[-1].attn.get_attn()  # shape: (B, Heads, N, N)

    # Use CLS token attention only
    attn = attn_maps[0, :, 0, 1:]  # (Heads, Tokens)
    attn_mean = attn.mean(0).reshape(14, 14).cpu().numpy()  # reshape to patch grid
    attn_mean = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min())

    plt.imshow(image.resize((224, 224)))
    plt.imshow(attn_mean, cmap='jet', alpha=0.6, extent=(0, 224, 224, 0))
    plt.title("ViT Attention Map (Last Layer)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vit_model()
    model.load_state_dict(torch.load("/checkpoints/best_vit.pth", map_location=device))

    # Sample image path
    image = Image.open("sample.jpg").convert("RGB")
    show_attention_map(model, image, device)
