import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from test_loop import accuracy_fn


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



def predict_unseen(model: torch.nn.Module, class_names: list[str], image_size: tuple, image_path: str, device: torch.device):


    # Open image
    img = Image.open(image_path)
    image_transform = transforms.Compose([transforms.Resize(image_size),
                                          transforms.ToTensor(),])

    ### Predict on image ###

    # Make sure the model is on the target device
    #model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        
        transformed_image = image_transform(img).unsqueeze(dim=0) # unsqueeze to account for batch size ([batch_size, color_channels, height, width])

        y_pred_logits = model(transformed_image.to(device))

        # Convert logits -> prediction probabilities -> labels (using torch.softmax() for multi-class classification)
        y_pred_probs = torch.softmax(y_pred_logits, dim=1) # for multiclass > 2
        y_pred_label = y_pred_probs.argmax(dim=1) # for multiclass > 2
        
        #y_pred_probs = torch.sigmoid(y_pred_logits).squeeze()
        #y_pred_label = torch.round(y_pred_probs)# its one value output, no need for argmax


    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(f"Prediction: {class_names[y_pred_label]} | Prob: {y_pred_probs.max():.3f}")
    plt.axis(False)
    #plt.savefig('cm/predict_unseen.png')
