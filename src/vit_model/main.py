import torch
import data_prep, model, train_test, predict, plots


if __name__ == "__main__":
    
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    
    # Data prep
    batch_size = 32
    train_dataloader, test_dataloader, y_train, y_test, class_names = data_prep.data_prep_pytorch_mnist(batch_size)
    
    # Build model ANN or CNN
    #model = model.Multi_Classification_ANN().to(device) # this can only work for gray_scale
    model.Multi_Classification_CNN(1, 10, 2).to(device) # in_channel is 1 for MNIST and 3 for achive data
    
    
    # Train and test
    ## - Setup hyperparameters
    num_epochs = 5
    lr = 0.001
    
    

    train_loss, test_loss, epoch_count = train_test.train_test(model, lr, num_epochs, train_dataloader, test_dataloader, device)
    
    
    
    ## - Plot loss
    plots.plot_loss(epoch_count, train_loss, test_loss)
    
    
    # Predict
    predict = predict.predict_test(model,
                                   test_dataloader)
    
    ## - Plot prediction
    plots.plot_predictions_cm(class_names, y_test, predict)
    
    
    predict_unseen_and_plot = predict.predict_unseen(model, class_names, image_path, device)
    
    # print parameter
    #print(model.state_dict())
    
    # save model
    #torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH) 
    
    
    # load model
    #loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    
    
    

