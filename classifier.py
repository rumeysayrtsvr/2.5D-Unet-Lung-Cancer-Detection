import torch
import torch.nn as nn
from torchvision import models

CLASSES = [
    "Adenocarcinoma",
    "Squamous Cell Carcinoma",
    "Normal"
]

def load_classifier(model_path, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
