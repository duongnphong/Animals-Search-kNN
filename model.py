# Task 1: Intitialize model with eval mode

from torchvision.models import *
import torch

def model_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    # model = nn.Sequential(*list(model.modules())[:-1])

    model.eval()
    
    return model


if __name__ == "__main__":
    model_eval()