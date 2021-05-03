import argparse
import torch
from torch import nn ,optim
from torch.nn import functional as F 
from torchvision import datasets,transforms as T,models
import numpy as np
import PIL
from PIL import Image
import cv2
import re 
from collections import OrderedDict



def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    #### Basic usage: python train.py data_directory
    parser.add_argument('dir', type=str, default='flowers/',help='data_directory')
    #### Options:
    parser.add_argument('save_dir', type=str,nargs='?',const='checkpoint.pth',default='checkpoint.pth',help='save_directory')
    parser.add_argument('arch', type=str,  nargs='?',const='vgg16', default='vgg16',help='model architectures')
    parser.add_argument('h1_units', type=int, nargs='?',const=256,default=256,help='hidden layer 1 units')
    parser.add_argument('h2_units', type=int, nargs='?',const=90 ,default=90,help='hidden layer 2 units')
    parser.add_argument('--device', action='store_true', default=False, help='Use GPU')
    parser.add_argument('lr', type=float, nargs='?',const=.001, default=.001,help='learning rate')
    parser.add_argument('epochs', type=int, nargs='?',const=15,default=15,help='learning rate')
    

    return parser.parse_args()

args=get_input_args()


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
##### def Image_datasets
def image_datasets(data_directory):
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = { 
    'train': [
        data_directory+'train',   
        T.Compose([
        T.RandomResizedCrop(224),
        T.RandomRotation(30),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean,std)])],
    'valid':[ 
        data_directory+'valid',
        T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean,std)])],
    'test': [
        data_directory+'test',
        T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean,std)])] 
    }
    
    image_datasets={}
    for mode in data_transforms:
        image_datasets[mode]=datasets.ImageFolder(data_transforms[mode][0],transform=data_transforms[mode][1])
    return image_datasets

#  Using the image datasets and the trainforms, define the dataloaders
def data_generators(image_datasets):
    data_loaders={}
    for mode in image_datasets:
        data_loaders[mode]= torch.utils.data.DataLoader(image_datasets[mode], batch_size=64, shuffle=True)
    return data_loaders 
image_datasets=image_datasets(args.dir)
dataloaders=data_generators(image_datasets)

import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
N_classes=len(cat_to_name.keys())

# TODO: Build and train your network
def model_generator(arch,h1,h2):
    if arch=='vgg16':
        model= models.vgg16(pretrained=True)
        ### Turn of the parameters of model
        for param in model.parameters():
            param.requires_grad = False
        ### Feedforward Classifier
 
        model.classifier=nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(25088, h1)), #hidden layer 1 sets output to 256
            ('relu1', nn.ReLU()),
            ('dropout',nn.Dropout(0.4)), #could use a different droupout probability,but 0.4 usually works well
            ('hidden_layer1', nn.Linear(h1, h2)), #hidden layer 1 output to 90
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(h2,N_classes)),#output size = 102 N_classes
            ('output', nn.LogSoftmax(dim=1))]))# For using NLLLoss()
        
    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
        for param in model.parameters():
            param.requires_grad = False
        ### Feedforward Classifier
        model.classifier=nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(1024, h1)), #hidden layer 1 sets output to 256
            ('relu1', nn.ReLU()),
            ('dropout',nn.Dropout(0.4)), #could use a different droupout probability,but 0.4 usually works well
            ('hidden_layer1', nn.Linear(h1, h2)), #hidden layer 1 output to 90
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(h2,N_classes)),#output size = 102 N_classes
            ('output', nn.LogSoftmax(dim=1))]))# For using NLLLoss()
            
    return model
model=model_generator(args.arch,args.h1_units,args.h1_units)
# Convert model to be used on GPU
device = torch.device('cuda' if torch.cuda.is_available() and args.device else 'cpu')
model.to(device)
    
    



#define the criterion and the optimizer to be used for training.
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
Train_Loss,Valid_Loss,Accuracy=[],[],[]
def train(dataloaders,epochs):
    for epoch in range(epochs):
        for mode in ['train','valid']:
            if mode == 'train':
                model.train()
                train_loss=0
                for images,labels in dataloaders[mode]: ### Loop about data
                    inputs,labels=images.to(device),labels.to(device)  ### move the data to GPU
                    logps=model.forward(inputs) ### apply feedfoward classifier
                    loss=criterion(logps,labels) ### return the loss
                    train_loss+=loss.item()     ### add loss to train_loss
                    #### Do Backward Step
                    optimizer.zero_grad()  # Clear the gradients
                    loss.backward()
                    optimizer.step()
                else :
                    train_loss/=len(dataloaders[mode])
                    Train_Loss.append(train_loss)
                
                    print(f"Epoch:=>  {epoch+1}/{epochs}.. "
                    f"Train loss:=>  {Train_Loss[-1]:.3f}.. ")
            elif mode =='valid':
                valid_loss=0
                acc = 0
                model.eval()
                with torch.no_grad():
                    for images,labels in dataloaders[mode]:
                        inputs,actual=images.to(device),labels.to(device)### move the data to GPU
                        logps=model.forward(inputs) ### apply feedfoward classifier
                        valid_loss+=criterion(logps,actual).item() ### return the loss
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, predicted = ps.topk(1, dim=1)
                        equals = predicted == actual.view(*predicted.shape)
                        acc+= torch.mean(equals.type(torch.FloatTensor)).item()
                    else :
                        valid_loss/=len(dataloaders[mode])
                        Valid_Loss.append(valid_loss)
                    
                        acc/=len(dataloaders[mode])
                        Accuracy.append(acc)
                    
                        print(f"Validation loss:=>  {Valid_Loss[-1]:.3f}.. "
                              f"Validation accuracy:=> {100*Accuracy[-1]:.3f}%",
                            "#"*68)
train(dataloaders,args.epochs)
## TODO: Do validation on the test set
#### https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Correct = 0
Total = 0
with torch.no_grad():
    for data in dataloaders['test']:
        images, labels = data
        inputs,labels=images.to(device),labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        Total  += labels.size(0)
        Correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images:=> {:.1f} %'.format(100 * Correct / Total))


# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoints={
    'N_classes':N_classes,
    'model':args.arch,
    'class_to_idx': model.class_to_idx ,
    'state_dict':model.state_dict(),
    'classifier':model.classifier,
    'optimizer' :optimizer,
    'epochs' : args.epochs,
    'learning rate':args.lr,
    'hidden-layers':[args.h1_units,args.h1_units]
    }
torch.save(checkpoints,'checkpoint.pth')
torch.save(checkpoints,args.save_dir)

