import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from src.vit_model.test_loop import accuracy_fn


def predict_test(model: torch.nn.Module, 
                test_dataloader: torch.utils.data.DataLoader,
                device: torch.device):

    """
    Generates predictions for the test dataset using a trained model.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model to be used for predictions.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) to perform the computations on.
        
    Returns:
        torch.Tensor: A tensor containing the predicted class indices for the test dataset.
    """
    
    y_preds = []
    # Put the model in evaluation mode
    model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:
            
            # 1. Send data to target device
            X_test, y_test = X.to(device), y.to(device)
            
            # 2. Forward pass on test data
            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            y_preds.append(test_pred.cpu())
            
    return torch.cat(y_preds)