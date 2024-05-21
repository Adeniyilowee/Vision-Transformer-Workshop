import torch
from tqdm.auto import tqdm

from torch import nn
import train_loop, test_loop


def train_test(model, lr, epochs, train_dataloader, test_dataloader, device):

    
    train_loss_ = []
    test_loss_ = []
    epoch_count = []
    
    train_acc_ = []
    test_acc_ = []
    
    # Set loss fucntion
    loss_fn = nn.CrossEntropyLoss() # for multi class (i.esout_features >1) and no need for sigmoid / softmax
    #loss_fn = nn.BCEWithLogitsLoss()
    # Create an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, 
                                 betas=(0.9, 0.999), weight_decay=0.3) # according to the paper
    
    
    for epoch in tqdm(range(epochs)):
        epoch_count.append(epoch)
        
        # Start training with help from train_loop.py
        train_loss, train_acc = train_loop.train_loop(model, loss_fn, optimizer, device, train_dataloader)
        train_loss_.append(train_loss)
        train_acc_.append(train_acc)
        
        # Start training with help from test_loop.py
        test_loss, test_acc = test_loop.test_loop(model, loss_fn, optimizer, device, test_dataloader)
        test_loss_.append(test_loss)
        test_acc_.append(test_acc)
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}")
    

    
    return train_loss_, test_loss_, epoch_count


