from skimage.color import rgb2lab
import os 
import random 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

class PortraitsDataset(Dataset):
    def __init__(self, colored_root_dir):
        """
        Args:
            colored_root_dir (string): Directory with all the colored images.
        """
        self.colored_root_dir = colored_root_dir
        self.colored_images_paths = os.listdir(self.colored_root_dir)

    def __len__(self):
        return len(self.colored_images_paths)

    def __getitem__(self, index):
        colored_image = self.colored_images_paths[index]
        colored_image_path = os.path.join(self.colored_root_dir, colored_image)
        image = cv2.imread(colored_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image,(256,256))
        image_np = np.array(image)
        image_gray = torch.tensor(rgb2lab(1.0/255*image_np)[:,:,0]).reshape(1,64,64)
        image_colored = torch.tensor(rgb2lab(1.0/255*image_np)[:,:,1:]).permute(2, 0,1)
        sample = {'colored_image': image_colored, 'gray_image': image_gray}
        return sample

    def viz_random(self):
        index = random.randint(0, self.__len__())
        sample = self.__getitem__(index)
        sample_in_gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
        _ , axs = plt.subplots(1,2)
        axs[0].imshow(sample)
        axs[1].imshow(sample_in_gray, cmap = plt.cm.gray)
        plt.show()

data = PortraitsDataset("pokemon_data")
print(data.colored_images_paths)
print(data.__getitem__(1))