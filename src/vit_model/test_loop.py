import torch
from torch import nn

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


def test_loop(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device, 
              test_dataloader: torch.utils.data.DataLoader):
    
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    
    """
    
    # Put the model in evaluation mode
    model.eval()
    
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            # Send data to target device
            X_test, y_test = X.to(device), y.to(device)
            
            # 1. Forward pass on test data
            test_logits = model(X_test)
            
            # 2. Caculate loss on test data
            loss = loss_fn(test_logits, y_test) #type(torch.float)
            test_loss += loss.item()
            
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1) # for multiclass > 2
            #test_pred = torch.round(torch.sigmoid(test_logits)) # its one value output, no need for argmax
            #test_acc += accuracy_fn(y_test, test_pred)
            test_acc += ((test_pred == y_test).sum().item()/len(test_logits))
            


            
            
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)     


    return test_loss, test_acc