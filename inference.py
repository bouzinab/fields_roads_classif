import torch
import os
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from data_utils import ImageDataset,get_loader, get_clip_embedding
from model import ClassificationHead
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a function to evaluate the test accuracy of the model.
def eval_model(data_loader):
    model.eval()

    total_epoch_loss = 0
    total_epoch_acc = 0
    ncorrect = ntotal = 0
    with torch.no_grad():
        for idx, data in tqdm(enumerate(data_loader)):
            
            input = data[0]
            input = input.to(device)
            output = model.forward(input)
            output = output.squeeze()

            target = data[1].float() 
            target = target.to(device)            
            loss =  criterion(output, target)

 
            ops=torch.sigmoid(output)
            predictions = (ops >= 0.5).float()
            total_epoch_loss += loss.item() 
            ntotal += len(target)
            ncorrect += torch.sum(predictions == target)

        total_epoch_acc = ncorrect.item() / ntotal
        return total_epoch_loss/(idx+1), total_epoch_acc

def main(root_dir, batch_size):
    transform = get_clip_embedding
    
    data = ImageDataset(root_dir, transform)
    batches = get_loader(data, batch_size)
    loss, acc = eval_model(batches)        
    return loss, acc

def predict_class(image):
    model.eval()

    input = get_clip_embedding(image)
    output = model.forward(input)
    ops=torch.sigmoid(output)
    prediction = (ops >= 0.5)

    cat = 'field'

    if prediction == 1:
        cat = 'road'

    return ops, prediction, cat


def predict_clip_only(image_path):

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = processor(text=["a photo of a field", "a photo of a road"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    prediction=torch.argmax(probs,dim=1)

    cat = 'field'
    if prediction == 1:
        cat = 'road'

    return probs, prediction, cat



#######################################################################################################
#######################################################################################################
if __name__ == '__main__':
    
    embed_dim = 512
    hidden_dim = 256 

    # Defining, the model, the gradient descent ##########################################################
    model = ClassificationHead(embed_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = 'model_checkpoint_2.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    #############################################################################################################
    #############################################################################################################

    """
    root_dir = "./dataset/test_images/"   #path to dataset
    images_file = [f for f in os.listdir(root_dir)]
    
    for image in images_file:
        image_path = root_dir + image
        ops, prediction, cat = predict_class(image_path)
        ops2, prediction2, cat2 = predict_clip_only(image_path)

        print('image', image_path)
        print('probs', ops)
        print('probs clip', ops2)

        print('prediction', prediction)
        print('prediction clip', prediction2)

        print('category', cat)
        print('category clip', cat2)

        print('--------------------------------')
    """
    #########################################################################################################
    #######################################################################################################
        
    parser = argparse.ArgumentParser(description="Process arguments for evaluation.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for data loader.")

    args = parser.parse_args()

    loss, acc = main(args.root_dir, args.batch_size)
    print('loss', loss)
    print('acc', acc)
    
    #python infence.py --root_dir ./path/to/your/dataset