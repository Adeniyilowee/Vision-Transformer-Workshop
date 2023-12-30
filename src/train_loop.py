import torch
from torch import nn


def train_loop(model: torch.nn.Module,
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device, 
               train_dataloader: torch.utils.data.DataLoader):
    
    
    # Put model in training mode (this is the default state of a model)
    model.train()
    
    train_loss, train_acc = 0, 0
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(train_dataloader):
        # Send data to target device
        X_train, y_train = X.to(device), y.to(device)

        # 1. Forward pass on train data using the forward() method inside 
        y_logits = model(X_train)
        
        
        # 2. Calculate the loss (how different are our models predictions to the ground truth)
        loss = loss_fn(y_logits, y_train) # y_logits because our loss function has sigmoid in-built
        train_loss += loss.item() # because we are dealing with batches
        

        # 3. Zero grad of the optimizer
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Progress the optimizer
        optimizer.step()
        
        
        # Calculate and accumulate accuracy metric across all batches
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        #y_pred = torch.round(torch.sigmoid(y_logits)) # its one value, no argmax needed

        
        # acc += accuracy_fn(y_train, y_pred) # replaced with below
        train_acc += (y_pred == y_train).sum().item()/len(y_logits)

        
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    return train_loss, train_acc
    
    
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc






