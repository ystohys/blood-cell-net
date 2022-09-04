from contextlib import redirect_stderr
import os
import numpy as np
import pandas as pd
import cv2

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