import os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from reference_detect.engine import train_one_epoch, evaluate
import reference_detect.utils as utils
from data_utils import BloodCellDataset
from bcnet import get_bcnet


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
        - train_idxs: List of indices corresponding to the training images
        - transforms: Composed transforms obtained from get_transform function
        - num_epochs: Number of epochs to train for
        - batch_size: Batch size to use for training
        - learning_rate: Initial learning rate

    Returns:
        - model: trained model
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

    Returns:
        - train_metrics: dict containing 3 mAP metrics and their histories when evaluated on training data
        - val_metrics: dict containing 3 mAP metrics and their histories when evaluated on validation data
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
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    # Implement learning rate scheduler to divide learning rate by 10 every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.1)
    val_metrics = {'coco_map':[], 'map_50':[], 'map_75':[]}
    train_metrics = {'coco_map':[], 'map_50':[], 'map_75':[]}
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_scheduler.step()
        tmp_val_met = evaluate(model, val_loader, device=device) # coco_eval object, take metrics from here
        val_metrics['coco_map'].append(tmp_val_met.coco_eval['bbox'].stats[0])
        val_metrics['map_50'].append(tmp_val_met.coco_eval['bbox'].stats[1])
        val_metrics['map_75'].append(tmp_val_met.coco_eval['bbox'].stats[2])
        with HiddenPrints():
            tmp_train_met = evaluate(model, no_transform_loader, device=device)
            train_metrics['coco_map'].append(tmp_train_met.coco_eval['bbox'].stats[0])
            train_metrics['map_50'].append(tmp_train_met.coco_eval['bbox'].stats[1])
            train_metrics['map_75'].append(tmp_train_met.coco_eval['bbox'].stats[2])
    return model, train_metrics, val_metrics


def kfcv_model(
    model_type,
    dir_name,
    train_idxs,
    train_transforms,
    val_transforms,
    num_epochs,
    batch_size,
    learning_rate,
    random_seed=42,
    k_folds=5
):
    """
    Performs k-fold cross validation using the given training set.

    Args:
        - model_type: a string representing the model type, one of ['frcnn', 'retina', 'ssd']
        - dir_name: (Relative) path to directory where the data is stored
        - train_idxs: Indices of images in training dataset
        - train_transforms: Composed transforms obtained from get_transform function for training data
        - val_transforms: Composed transforms obtained from get_transform function for validation data
        - num_epochs: Number of epochs to train for
        - batch_size: Batch size to use for training
        - learning_rate: Initial learning rate
        - random_seed: Random seed used to generate the k-folds, set to fix folds across each test
        - k_folds: Number of folds in k-fold cross validation

    Returns:
        - total_train_hist: dict containing 3 mAP metrics and their histories for each fold 
                            when evaluated on training data
        - total_val_hist: dict containing 3 mAP metrics and their histories for each fold
                          when evaluated on validation data
    """
    kf_splitter = KFold(n_splits=k_folds, random_state=random_seed, shuffle=True)
    total_train_hist = {'coco_map': [], 'map_50': [], 'map_75': []}
    total_val_hist = {'coco_map': [], 'map_50': [], 'map_75': []}
    for fold_num, train_val_idx in enumerate(kf_splitter.split(train_idxs)):
        print('Fold {0}'.format(fold_num+1))
        sub_train_idx = np.array(train_idxs)[train_val_idx[0]]
        sub_val_idx = np.array(train_idxs)[train_val_idx[1]]
        sub_train_idx = sub_train_idx.tolist()
        sub_val_idx = sub_val_idx.tolist()
        tmp_model = get_bcnet(model_type, 3, 1)
        _, fold_train_metrics, fold_val_metrics = hocv_model(
            tmp_model,
            dir_name,
            sub_train_idx,
            sub_val_idx,
            train_transforms,
            val_transforms,
            num_epochs,
            batch_size,
            learning_rate
        )
        for i in total_train_hist.keys():
            total_train_hist[i].append(fold_train_metrics[i])
            total_val_hist[i].append(fold_val_metrics[i])

    for s in total_train_hist.keys():
        # convert all histories to numpy arrays within the dictionary
        total_train_hist[s] = np.array(total_train_hist[s])
        total_val_hist[s] = np.array(total_val_hist[s])
    return total_train_hist, total_val_hist


def plot_metrics_per_epoch(
    train_hist,
    val_hist,
    title=None
):
    """
    Function to plot the training and validation COCO-mAP scores throughout training.
    """
    if len(train_hist['coco_map'].shape) == 1: # For holdout cv
        plt.plot(train_hist['coco_map'])
        plt.plot(val_hist['coco_map'])
        legends = ['train', 'val']
        if title:
            plt.title(title)
        plt.ylabel('COCO-mAP score')
        plt.xlabel('Epoch')
        plt.legend(legends, loc='upper left')
    else:
        fig, ax = plt.subplots()
        train_map_mean = np.mean(train_hist['coco_map'], axis=0)
        train_map_std = np.std(train_hist['coco_map'], axis=0)
        tp1 = ax.plot(np.arange(len(train_map_mean)), train_map_mean)
        tp2 = ax.fill(np.NaN, np.NaN, tp1[0].get_color(), alpha=0.5)
        ax.fill_between(
            x=np.arange(len(train_map_mean)),
            y1=train_map_mean+train_map_std,
            y2=train_map_mean-train_map_std,
            alpha=0.5
        )
        val_map_mean = np.mean(val_hist['coco_map'], axis=0)
        val_map_std = np.std(val_hist['coco_map'], axis=0)
        vp1 = ax.plot(np.arange(len(val_map_mean)), val_map_mean)
        vp2 = ax.fill(np.NaN, np.NaN, vp1[0].get_color(), alpha=0.5)
        ax.fill_between(
            x=np.arange(len(val_map_mean)),
            y1=val_map_mean+val_map_std,
            y2=val_map_mean-val_map_std,
            alpha=0.5
        )
        legend_obj = [(tp2[0], tp1[0]), (vp2[0], vp1[0])]
        legends = ['train', 'val']
        ax.set_ylim(0,1)
        if title:
            ax.set_title(title)
        ax.set_ylabel('COCO-mAP score')
        ax.set_xlabel('Epoch')
        ax.legend(legend_obj, legends, loc='upper left')
        plt.show()


def eval_model(
    model,
    dir_name,
    test_idxs,
    test_transforms,
    batch_size=1
):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    test_dataset = BloodCellDataset(dir_name, test_transforms)
    test_loader = DataLoader(
        dataset=test_dataset,
        sampler=SubsetRandomSampler(test_idxs),
        batch_size=batch_size,
        collate_fn=utils.collate_fn
    )
    test_met = {}
    # No need to set model.eval() because it is done within evaluate function
    tmp = evaluate(model, test_loader, device=device)
    test_met['coco_map'] = tmp.coco_eval['bbox'].stats[0]
    test_met['map_50'] = tmp.coco_eval['bbox'].stats[1]
    test_met['map_75'] = tmp.coco_eval['bbox'].stats[2]
    return test_met


def model_predict(model, img):
    if model.training:
        model.eval()
    prediction = model(img)
    model.train()
    return prediction

        
def save_model(model, model_name=None, save_path=None):
    if not save_path:
        if not model_name:
            raise ValueError("model_name must be provided if no save_path given!")
        save_path = '{0}.pth'.format(model_name)
    torch.save(model.state_dict(), save_path)
    print('Model saved at: {0}'.format(save_path))


def load_model(model_obj, saved_path):
    model_obj.load_state_dict(torch.load(saved_path))

