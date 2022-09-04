import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import cv2


def plot_img_w_box(img, tgt):
    draw_img = ImageDraw.Draw(img)
    for i, box in enumerate(tgt['boxes']):
        if tgt['labels'][i] == 1:
            draw_img.rectangle(box, outline='red')
            font = ImageFont.truetype("sans_serif.ttf", 8)
            draw_img.text(xy=(box[0]-5, box[1]-5), text="RBC", fill="red", font=font)
        elif tgt['labels'][i] == 2:
            draw_img.rectangle(box, outline='blue')
            font = ImageFont.truetype("sans_serif.ttf", 8)
            draw_img.text(xy=(box[0]-5, box[1]-5), text="WBC", fill="blue", font=font)
    img.show()


def show_img_with_anno(img_file_path, anno_file_path):
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