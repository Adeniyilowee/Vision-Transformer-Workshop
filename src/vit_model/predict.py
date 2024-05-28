import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from src.vit_model.test_loop import accuracy_fn


def predict_test(model: torch.nn.Module, 
                test_dataloader: torch.utils.data.DataLoader,
                device: torch.device):
    
    y_preds = []
    # Put the model in evaluation mode
    model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:
            # Send data to target device
            X_test, y_test = X.to(device), y.to(device)
            
            # 1. Forward pass on test data
            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1) # for multiclass > 2
            #test_pred = torch.round(torch.sigmoid(test_logits))
            y_preds.append(test_pred.cpu())
        

           
    return torch.cat(y_preds)