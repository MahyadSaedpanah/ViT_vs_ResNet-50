import torch.nn as nn
import torchvision.models as models

def get_resnet_model(num_classes=100, pretrained=True):
    # Load ResNet-50 from torchvision
    model = models.resnet50(pretrained=pretrained)

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

if __name__ == "__main__":
    model = get_resnet_model()
    print(model)
