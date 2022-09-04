import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as TT
import reference_detect.transforms as T


class BloodCellDataset(Dataset):
    def __init__(self, dir_path, transforms=None):
        self.dir_path = dir_path
        self.all_images = [i for i in sorted(os.listdir(os.path.join(dir_path, 'images'))) if i.endswith('.png')]
        self.transforms = transforms
        self.anno_df = pd.read_csv(os.path.join(self.dir_path, 'clean_anno.csv'))
    
    def __get__item(self, idx):
        img_id = self.all_images[idx]
        img_path = os.path.join(self.dir_path, "images", img_id)
        img = Image.open(img_path).convert("RGB")  
        # The image pixels will range from [0, 255] so we will normalize using one of the transforms later
        boxes = []
        labels = []
        bound_box_df = self.anno_df.loc[self.anno_df['image']==img_id, ['xmin', 'ymin', 'xmax', 'ymax', 'label']]
        bound_box_list = bound_box_df.to_dict('records')
        for bb in bound_box_list:
            xmin, xmax = int(round(bb['xmin'])), int(round(bb['xmax']))
            ymin, ymax = int(round(bb['ymin'])), int(round(bb['ymax']))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(bb['label'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((boxes.shape[0], ), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['image_id'] = image_id

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.all_images)


class PILToNormTensor(nn.Module):
    """
    Custom transformation to normalize PIL image values before converting to torch.Tensor
    """
    def forward(
        self, 
        image,
        target
    ):
        image = F.to_tensor(image)
        return image, target


class RandomVerticalFlip(TT.RandomVerticalFlip):
    def forward(
        self, image, target
    ):
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height, _ = F.get_dimensions(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(PILToNormTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return T.Compose(transforms)


################################################
## Functions for data cleaning and statistics ##
################################################

def get_duplicates(anno_df):
    if isinstance(anno_df, str):
        anno_df = pd.read_csv(anno_df)
    dup_df = anno_df.loc[anno_df.duplicated(subset=['image','xmin','ymin','xmax','ymax'], keep=False),:]
    return dup_df


def get_class_distribution(anno_df, agg_idx, plot_dpi=130):
    if isinstance(anno_df, str):
        anno_df = pd.read_csv(anno_df)
    agg_df = anno_df.groupby(agg_idx, as_index=False).size()
    agg_df['percent'] = agg_df['size'] / agg_df.groupby(agg_idx[0])['size'].transform('sum')
    wide_df = agg_df.pivot(index=agg_idx[0], columns=agg_idx[1], values='percent')
    fig = plt.figure(dpi=plot_dpi)
    wide_df.plot(
        kind='bar', 
        stacked=True, 
        xticks=np.arange(0,100,step=10), 
        ylabel='Proportion of total annotated blood cells in image',
        color=['red','blue'],
        ax=plt.gca()
        )
    plt.show()