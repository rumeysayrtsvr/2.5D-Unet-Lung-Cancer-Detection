import torch
from networks import DenseUNet

def load_model(model_path, device):
    model = DenseUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
