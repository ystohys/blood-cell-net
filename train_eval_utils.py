import os, sys
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from reference_detect.engine import train_one_epoch, evaluate
import reference_detect.utils as utils
from data_utils import BloodCellDataset


class HiddenPrints:
    """
    Helper function to suppress print outputs.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def train_model(
    model,
    dir_name,
    train_idxs,
    transforms,
    num_epochs,
    batch_size,
    learning_rate
):
    """
    Trains a model using the given training set.

    Args:
        - model: One of the pre-trained models for object detection in torchvision, obtained from bcnet.py
        - dir_name: (Relative) path to directory where the data is stored
        - train_idxs: List of filenames of images in training set. E.g. ['image-1.png', 'image-2.png']
        - transforms: Composed transforms obtained from get_transform function
        - num_epochs: Number of epochs to train for
        - batch_size: Batch size to use for training
        - learning_rate: Initial learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bc_dataset = BloodCellDataset(dir_name, transforms)
    
    train_loader = DataLoader(
        dataset=bc_dataset,
        sampler=SubsetRandomSampler(train_idxs), # Using SubsetRandomSampler to sample randomly from training indices
        batch_size=batch_size,
        collate_fn=utils.collate_fn
        )
    model.to(device)
    # Build optimizer
    params = [p for p in model.parameters()]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    # Implement learning rate scheduler to divide learning rate by 10 every 5 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
    return model


def hocv_model(
    model,
    dir_name,
    train_idxs,
    val_idxs,
    train_transforms,
    val_transforms,
    num_epochs,
    batch_size,
    learning_rate
):
    """
    Performs hold-out cross validation using the given training and validation indices.

    Args:
        - model: One of the pre-trained models for object detection in torchvision, obtained from bcnet.py
        - dir_name: (Relative) path to directory where the data is stored
        - train_idxs: Indices of images in training dataset
        - val_idxs: Indices of images in validation dataset
        - train_transforms: Composed transforms obtained from get_transform function for training data
        - val_transforms: Composed transforms obtained from get_transform function for validation data
        - num_epochs: Number of epochs to train for
        - batch_size: Batch size to use for training
        - learning_rate: Initial learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = BloodCellDataset(dir_name, train_transforms)
    non_augment_train = BloodCellDataset(dir_name, val_transforms)
    val_dataset = BloodCellDataset(dir_name, val_transforms)
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=batch_size,
        collate_fn=utils.collate_fn
        )
    no_transform_loader = DataLoader(
        dataset=non_augment_train,
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=batch_size,
        collate_fn=utils.collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=SubsetRandomSampler(val_idxs),
        batch_size=batch_size,
        collate_fn=utils.collate_fn
    )
    model.to(device)
    # Build optimizer
    params = [p for p in model.parameters()]
    optimizer = torch.optim.SGD(params, lr=learning_rate)
    # Implement learning rate scheduler to divide learning rate by 10 every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    val_metrics = []
    train_metrics = []
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        tmp_val_met = evaluate(model, val_loader, device=device) # coco_eval object, take metrics from here
        val_metrics.append(tmp_val_met.coco_eval['bbox'].stats[0:3])
        with HiddenPrints():
            tmp_train_met = evaluate(model, no_transform_loader, device=device)
            train_metrics.append(tmp_train_met.coco_eval['bbox'].stats[0:3])
    return model, np.array(val_metrics), np.array(train_metrics)

