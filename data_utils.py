import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

def get_clip_embedding(image_path):
    """
    Takes an image path as input, loads the pretrained CLIP model
    and computes the image embedding using the model. The embedding dim is 512

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: A tensor containing the image embedding.
    """    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")              
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")          
                                                                                    
    image = Image.open(image_path)
    inputs = processor(text=[""], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
 
    return outputs.image_embeds


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=get_clip_embedding, task="training"):
        self.root_dir = root_dir
        self.task = task
        if self.task == "training":
            self.classes = ['fields', 'roads']
            self.class_to_label = {class_name: idx for idx, class_name in enumerate(self.classes)}
            self.image_paths, self.labels = self._get_image_paths_and_labels()
        self.transform = transform
        

    def _get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                image_paths.append(image_path)
                labels.append(self.class_to_label[class_name])
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image_path)

        return image, label
    

def get_loader(
        dataset,
        batch_size
):

    data_loader = DataLoader(
        dataset= dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return data_loader


if __name__ == '__main__':
    data = ImageDataset(root_dir="./dataset")

    print(len(data))
    print(data[0][0].shape)
    print('image embedding', data[0][0])
    print('image label', data[0][1])


    batch = get_loader(data, 3)
    print(len(batch))        

