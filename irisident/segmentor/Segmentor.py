import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
import sys
from pathlib import Path
import yaml
import torch




class Segmentor: 
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
    def crop(self):  
        res = self.model.predict(source=self.source,imgsz=self.imgsz,device=self.device,half=self.half,classes=self.classes)
    # iterate detection results
        for r in res:
            img = np.copy(r.orig_img)
            img_name = Path(r.path).stem
            # iterate each object contour
            c=r[0]
            if len(c.masks.xy[0])>0:
                b_mask = np.zeros(img.shape[:2], np.uint8)
                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # Choose one:

                # OPTION-1: Isolate object with black background
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, img)

                # OPTION-2: Isolate object with transparent background (when saved as PNG)
                #isolated = np.dstack([img, b_mask])

                # OPTIONAL: detection crop (from either OPT1 or OPT2)
                x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                iso_crop = isolated[y1:y2, x1:x2]

                # TODO your actions go here
                filename = img_name+'.png'
                filename = os.path.join(self.outdir,filename)

                cv2.imwrite(filename, iso_crop)
                print('savemask:',filename)
