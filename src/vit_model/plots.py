import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def plot_predictions_cm(class_names, y_test, predict):


    # 2. Setup confusion matrix instance and compare predictions to targets
    cm = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    cm_tensor = cm(preds=predict, target=y_test)

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm_tensor.numpy(), 
                                    class_names=class_names, 
                                    figsize=(10, 7))
    
    #fig.savefig('cm/pred_cm.png')

    

def plot_loss(epoch_count, train_loss, val_loss):
    # Plot Test and Train Loss
    plt.plot(epoch_count, train_loss, label="Train loss")
    plt.plot(epoch_count, val_loss , label="Validation loss")
    plt.title("Training and Validation loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    #plt.savefig('loss.png');