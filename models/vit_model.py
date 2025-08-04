import torch.nn as nn
import timm

def get_vit_model(num_classes=100, pretrained=True):
    # Load ViT-Base-16 from timm
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

    # Replace classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model

if __name__ == "__main__":
    model = get_vit_model()
    print(model)
