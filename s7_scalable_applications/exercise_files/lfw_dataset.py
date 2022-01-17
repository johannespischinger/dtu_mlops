"""
LFW dataloading
"""
import argparse
import time
import os
import glob

import numpy as np
import torch
import torchvision.utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.images = []
        self.image_folders = [os.path.join(path_to_folder,folder) for folder in os.listdir(path_to_folder) if os.path.isdir(os.path.join(path_to_folder,folder)) is True]
        #self.images = [os.listdir(os.path.join(path_to_folder,folder)) for folder in os.listdir(path_to_folder) if os.path.isdir(os.path.join(path_to_folder,folder)) is True]
        for name in self.image_folders:
            for picture in os.listdir(name):
                self.images.append(os.path.join(name,picture))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img = Image.open(self.images[index],'r')
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw', type=str)
    parser.add_argument('-num_workers', default=1, type=list)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type= int)
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    #dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
    #                       num_workers=args.num_workers)
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        imgs = next(iter(dataloader))
        #fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        #for i, img in enumerate(imgs):
        #    img = img.detach()
        #    img = transforms.functional.to_pil_image(img)
        #    axs[0, i].imshow(np.asarray(img))
        #    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        #pass
        grid = torchvision.utils.make_grid(imgs)
        # reshape and plot (because MPL needs channel as the last dimension)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

    if args.get_timing:
        # lets do so repetitions
        std = []
        mean = []
        for workers in range(1,9):
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False,
                       num_workers=workers)
            res = []
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)
            mean.append(np.mean(res))
            std.append(np.std(res))
            print(f'Workers: {workers} - Timing: {np.mean(res)}+-{np.std(res)}')

        plt.errorbar(range(1,9),mean,std)
