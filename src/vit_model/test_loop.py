import torch
from torch import nn


def test_loop(model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device, 
              test_dataloader: torch.utils.data.DataLoader):

    """
    Evaluate the model on the test dataset and calculate the average loss and accuracy.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        loss_fn (torch.nn.Module): The loss function used to evaluate the model's performance.
        optimizer (torch.optim.Optimizer): The optimizer used during training (not used in this function, but typically part of the loop structure).
        device (torch.device): The device (CPU or GPU) to run the evaluation on.
        test_dataloader (torch.utils.data.DataLoader): The DataLoader providing the test dataset.

    Returns:
        float: The average loss on the test dataset.
        float: The average accuracy on the test dataset.
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


def accuracy_fn(y_true, y_pred):

    """
    Computes the accuracy of predictions.

    This function calculates the accuracy by comparing the predicted labels
    (`y_pred`) with the true labels (`y_true`). The accuracy is computed as
    the percentage of correct predictions.

    Args:
        y_true (torch.Tensor): The ground truth labels.
        y_pred (torch.Tensor): The predicted labels.

    Returns:
        float: The accuracy of the predictions, as a percentage.
    """
    
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc