import os
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import pandas as pd
import cv2
import torchvision
from torchvision.transforms.functional import to_pil_image


def plot_img_w_box(img, tgt, nms=False, iou_threshold=0.4):
    """
    Main function used to plot images and the respective bounding boxes. Images and bounding box 
    information fed to this must be obtained from a BloodCellDataset instance.

    Args:
    - img: An image (PIL image or torch.Tensor) obtained from a BloodCellDataset instance
    - tgt: Either predictions from the model or a targets obtained from a BloodCellDataset instance
    - nms: Whether to apply non-maximum suppression to the predictions. False by default
    - iou_threshold: IoU threshold to use when applying non-maximum suppresion
    """
    if isinstance(img, torch.Tensor):
        img = to_pil_image(img)
    if nms:
        tgt = non_max_suppress(tgt, iou_threshold)
    draw_img = ImageDraw.Draw(img)
    for i, box in enumerate(tgt['boxes']):
        if tgt['labels'][i] == 1:
            draw_img.rectangle((box[0],box[1],box[2],box[3]), outline='red')
            font = ImageFont.load_default()
            draw_img.text(xy=(box[0]-5, box[1]-5), text="RBC", fill="red", font=font)
        elif tgt['labels'][i] == 2:
            draw_img.rectangle((box[0],box[1],box[2],box[3]), outline='blue')
            font = ImageFont.load_default()
            draw_img.text(xy=(box[0], box[1]-10), text="WBC", fill="blue", font=font)
    img.show()


def non_max_suppress(pred, iou_threshold=0.4):
    """
    Performs non-maximum suppression to only keep the bounding boxes with the highest scores
    and remove those that overlap with them.

    Args:
    - pred: predictions output obtained directly from the model
    - iou_threshold: Any bounding box that has a lower confidence score than another bounding box, but
    has an IoU with that box that is greater than this threshold will be removed
    
    Return:
    - final_pred: prediction containing the remaining bounding boxes after NMS is performed
    """
    keep_arr = torchvision.ops.nms(pred['boxes'], pred['scores'], iou_threshold)
    final_pred = pred
    final_pred['boxes'] = final_pred['boxes'][keep_arr]
    final_pred['scores'] = final_pred['scores'][keep_arr]
    final_pred['labels'] = final_pred['labels'][keep_arr]
    return final_pred


def show_img_with_anno(img_file_path, anno_file_path):
    """
    Another function to show images and their bounding boxes in a new window. This function
    only accepts paths to the directory containing the images and the metadata csv file (annotations.csv)
    """
    img_file = os.path.split(img_file_path)[1]
    anno_df = pd.read_csv(anno_file_path)
    
    bound_box_df = anno_df.loc[anno_df['image']==img_file, ['xmin', 'ymin', 'xmax', 'ymax', 'label']]
    bound_box_list = bound_box_df.to_dict('records')
    
    img = cv2.imread(img_file_path)
    for bb in bound_box_list:
        xmin, xmax = int(round(bb['xmin'])), int(round(bb['xmax']))
        ymin, ymax = int(round(bb['ymin'])), int(round(bb['ymax']))
        clr = (0,0,255) if bb['label'] == 'rbc' else (255,0,0)
        img = cv2.rectangle(img, 
                            (xmin, ymin), 
                            (xmax, ymax), 
                            clr, 
                            2)
        cv2.putText(img, bb['label'], (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)

    cv2.imshow(img_file, img)
    cv2.waitKey(0)
    cv2.destroyWindow(img_file)