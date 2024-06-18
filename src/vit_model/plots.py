import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


def plot_predictions_cm(class_names, y_test, predict):

    """
    Plots the confusion matrix for given predictions and true labels.

    Args:
        class_names (list of str): Names of the classes.
        y_test (torch.Tensor): True labels.
        predict (torch.Tensor): Predicted labels.

    Returns:
        None
    """

    # 1. Setup confusion matrix instance and compare predictions to targets
    cm = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    cm_tensor = cm(preds=predict, target=y_test)

    # 2. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm_tensor.numpy(),
                                    class_names=class_names,
                                    figsize=(10, 7))


def plot_loss(epoch_count, train_loss, val_loss):

    """
    Plots the training and validation loss curves.

    Args:
        epoch_count (list of int): List of epoch numbers.
        train_loss (list of float): Training loss values for each epoch.
        val_loss (list of float): Validation loss values for each epoch.

    Returns:
        None
    """

    # Plot Test and Train Loss
    plt.plot(epoch_count, train_loss, label="Train loss")
    plt.plot(epoch_count, val_loss, label="Validation loss")
    plt.title("Training and Validation loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()


def plot_task(image1, image2):

    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot image 1
    axes[0].imshow(np.transpose(image1, (1, 2, 0)), cmap='viridis')
    axes[0].set_title('Forest')

    # Plot image 2
    axes[1].imshow(np.transpose(image2, (1, 2, 0)), cmap='viridis')
    axes[1].set_title('Mountain')

    # Remove axis ticks
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
