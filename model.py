import torch
import os
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from data_utils import ImageDataset, get_loader, get_clip_embedding
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim import lr_scheduler 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, src):
        src = src.to(device)
        output = self.fc1(src)
        output = self.relu(output)
        output = self.fc2(output)

        return output


# a function to train the model on 1 epoch.
def train_model(data_loader,epoch):

    model.train()
    total_loss = 0.0
    
    losses = []
    accuracies = []
    ncorrect = ntotal = 0
    log_interval= int(len(data_loader)/2)
    for idx, data in enumerate(data_loader): 
        
        optimizer.zero_grad()
        input = data[0].to(device) 

        output = model.forward(input) 
        output = output.squeeze()
        
        target = data[1].float()
        target = target.to(device)
        loss =  criterion(output, target) 
        ops=torch.sigmoid(output)
        predictions = (ops >= 0.5).float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent exploding gradient 
    
        optimizer.step()

        total_loss += loss.item() 
        ntotal += len(target)
        ncorrect += torch.sum(predictions == target)
        if idx % log_interval == 0 and idx > 0: #print the result after -log_interval- batches 
            cur_loss = total_loss / log_interval
            cur_acc = ncorrect.item() / ntotal

            print(
                "| epoch {:3d} | {:5d}/{:5d} steps | "
                "loss {:5.5f} | acc {:8.3f}".format(
                    epoch, idx, len(data_loader), cur_loss, cur_acc,
                )
            )
            losses.append(cur_loss)
            accuracies.append(cur_acc)
            total_loss = 0   
            ntotal = 0
            ncorrect = 0
             

    return losses, accuracies


# Train the model with a early_stopping, patience option. 
def experiment(data_loader_train, num_epochs = 5, early_stopping = True, patience=20, checkpoint_path="best_model.pth",log_dir="runs/training_new"):

    writer = SummaryWriter(log_dir=log_dir)

    train_losses = []
    train_acc = []
    if early_stopping: 
        best_loss = float("inf")
        counter = 0 
    print("Beginning training...")
    for epoch in range(1,num_epochs+1):
        losses, accuracies = train_model(data_loader_train,epoch)

        loss = np.mean(losses)
        acc = np.mean(accuracies)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        
        train_losses.append(loss) 
        train_acc.append(acc)   
        
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Learning Rate", current_lr, epoch)
        scheduler.step()   
        
        if early_stopping:
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), checkpoint_path)
                print('checkpoint saved.')
                counter= 0
            else:
                counter+= 1
                if counter==patience:
                    print(f"Validation loss did not improve for {patience} epochs. Early stopping.")
                    break  
    
    return train_losses, train_acc


#######################################################################################################
#######################################################################################################
if __name__ == '__main__':
    
    embed_dim = 512
    hidden_dim = 256 

    # Defining, the model, loss, the gradient descent and lr ##########################################################
    model = ClassificationHead(embed_dim, hidden_dim)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    lr = 0.0003  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    #############################################################################################################
    #############################################################################################################


    root_dir = "./dataset"   #path to dataset
    transform = get_clip_embedding
    task="training"
    
    batch_size = 3     
    num_epochs = 50

    early_stopping = True
    patience=15
    checkpoint_path="model_checkpoint_3.pth"
    log_dir="runs/training_classif_head_3"

    data = ImageDataset(root_dir, transform, task)
    train_batches = get_loader(data,batch_size)
        
    print('number of train batches ', len(train_batches))


    #########################################################################################################
    #######################################################################################################


    print('#################### training the model ############################################')

    train_losses, train_acc = experiment(train_batches, num_epochs, early_stopping, patience, checkpoint_path,log_dir)