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
import matplotlib.pyplot as plt 
from collections import OrderedDict
import json

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # #### Basic usage: python predict.py /path/to/image checkpoint
    parser.add_argument('image', type=str,default='flowers/test/10/image_07090.jpg',help=" path to image")
    parser.add_argument('checkpoint', type=str,default='checkpoint.pth',help='checkpoint')
    #### Options:
    parser.add_argument('TK', type=int,nargs='?', const=5, default=5,help='top classes')
    parser.add_argument('names', type=str, nargs='?',const='cat_to_name.json', default='cat_to_name.json',help=" category names")
    parser.add_argument('--device', action='store_true', default=False, help='Use GPU')
    return parser.parse_args()
args=get_input_args()
device = torch.device('cuda' if torch.cuda.is_available() and args.device else 'cpu')

with open(args.names, 'r') as f:
    cat_to_name = json.load(f)
N_classes=len(cat_to_name.keys())

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
data_transforms = { 
    'test': [
        'flowers/test',
        T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean,std)])] 
    }



#a function that loads a checkpoint and rebuilds the model
def load(filepath):
    ### that loads a checkpoint
    state=torch.load(filepath)
    ### rebuilds the mode
    if state['model']=='vgg16':
        model= models.vgg16(pretrained=True)
        ### Turn of the parameters of model
        for param in model.parameters():
            param.requires_grad = False
        ### Feedforward Classifier
        model.classifier=state['classifier']
    elif state['model']== 'Densenet':
        model = models.densenet121(pretrained=True)
        # Only train the classifier parameters, feature parameters are frozen
        for param in model.parameters():
            param.requires_grad = False
        ### Feedforward Classifier
        model.classifier=state['classifier']
    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    return model

model=load(args.checkpoint)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #### Read the Image
    img = Image.open(image)
    ### Apply the Test Transforms on img
    img= data_transforms['test'][1](img)
    ### Converts to Array
    img_array = np.array(img)
    ### The color channel needs to be first and retain the order of the other two dimensions.
    image = img_array.transpose((0, 1, 2))
    return image

def predict(image_path, model, Topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    
    image = process_image(image_path)
    #### Converts to Tensor
    image = torch.from_numpy(image)
    ### Add Dimension
    X= torch.unsqueeze(image, 0)
    with torch.no_grad():
        log_probs = model.forward(X.to(device))
   # Convert to linear scale
    prob = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = prob.topk(Topk,dim=1)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]

    # Convert to classes
    idx_to_class={}
    for key, val in model.class_to_idx.items():
        idx_to_class[val]=key

    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

image_path =args.image
# Set up title
pattern = '(\d+)\/'
flower_num =re.findall(pattern,image_path)
Title=cat_to_name[flower_num[0]]
print(f"Actual :=> {Title}")
print("Predictions :=>")

print(predict(image_path, model,args.TK))