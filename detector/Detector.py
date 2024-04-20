import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from PIL import Image
import argparse
import sys
from pathlib import Path
import yaml
import torch
import math

def get_rect(img_box):
    height, width = img_box.shape[:2]
    xc = int(width / 2)
    yc = int(height / 2)
    r = int(width / 2)
    p = 0.4

    x1l = int(r * (1 - math.sqrt(1 - math.pow(p, 2))))
    y1l = yc - int(r * p)
    if (img_box[y1l, x1l] == [0, 0, 0]).all():
        Xs = np.nonzero(np.any(img_box[y1l, :], axis=1))
        x1l = int(Xs[0][0])

    x2l = xc - int(r * p)
    y2l = yc + int(r * p)
    if (img_box[y2l, x1l] == [0, 0, 0]).all():
        Xs = np.nonzero(np.any(img_box[y2l, :], axis=1))
        x1l = int(Xs[0][0])

    x1r = xc + int(r * p)
    y1r = yc - int(r * p)
    x2r = int(r * (1 + math.sqrt(1 - math.pow(p, 2))))
    y2r = yc + int(r * p)
    if (img_box[y1r, x2r] == [0, 0, 0]).all():
        Xs = np.nonzero(np.any(img_box[y1r, :], axis=1))
        x2r = int(Xs[0][-1])
    if (img_box[y2r, x2r] == [0, 0, 0]).all():
        Xs = np.nonzero(np.any(img_box[y2r, :], axis=1))
        x2r = int(Xs[0][-1])

    return ((x1l, y1l), (x2l, y2l), (x1r, y1r), (x2r, y2r))



class Detector: 
    def __init__(self,args): 
            self.source=args.source
            self.weights=args.weights
            self.conf=args.conf
            self.imgsz=args.imgsz
            self.outdir=args.outdir
            self.half=False
            self.device='cpu'
            self.model=YOLO(self.weights)
            self.classes=args.iris_id
            self.model.to(self.device)
            if not os.path.exists(self.outdir): 
                os.makedirs(self.outdir)
    def crop(self):  
        res = self.model.predict(source=self.source,imgsz=self.imgsz,device=self.device,half=self.half,classes=self.classes)
    # iterate detection results
        for i, r in enumerate(res):
            boxes = r.boxes.cpu().numpy()
            # print(boxes)
            im_array = r.plot(labels=False, conf=False, boxes=False, masks=False)
            # Convert the BGR image to RGB.
            im = Image.fromarray(im_array[..., ::-1])
            # Get xyxy of bounding box
            box = boxes.xyxy[0]
            #boxhw = boxes.xywh[0]
            #print(boxhw)
            # Cropping logic here
            imcrop = im.crop((box[0], box[1], box[2], box[3]))
            iso_crop = cv2.cvtColor(np.array(imcrop), cv2.COLOR_RGB2BGR)
            (w,h) = iso_crop.shape[:2]
            b_mask = np.zeros(iso_crop.shape[:2], np.uint8)
            center=(int(h/2),int(w/2))
            axes=(int(h/2),int(w/2))
            cv2.ellipse(b_mask,center,axes,0,0,360,255,-1)
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, iso_crop)
            coords = get_rect(isolated)
            x1,y1 = coords[0]
            x2,y2 = coords[1]
            rectl_crop = iso_crop[y1:y2, x1:x2]
            x1,y1 = coords[2]
            x2,y2 = coords[3]
            rectr_crop = iso_crop[y1:y2, x1:x2]
            # TODO your actions go here
            img_name = Path(r.path).stem
            filename = img_name+'.png'
            filename_iso = img_name+'_iso.png'
            filename_l = img_name+'_l.png'
            filename_r = img_name+'_r.png'
            filename = os.path.join(self.outdir,filename)
            filename_l = os.path.join(self.outdir,filename_l)
            filename_r = os.path.join(self.outdir,filename_r)
            filename_iso = os.path.join(self.outdir,filename_iso)
            imcrop.save(filename, "png")
            #cv2.imwrite(filename, iso_crop)
            cv2.imwrite(filename_iso, isolated)
            cv2.imwrite(filename_l, rectl_crop)
            cv2.imwrite(filename_r, rectr_crop)
            print('savecrop:',filename)
            print('save left rect:',filename_l)
            print('save right rect:',filename_r)
