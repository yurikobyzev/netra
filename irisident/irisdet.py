import cv2
import os
from detector.Detector import Detector
import argparse
import datetime

parser = argparse.ArgumentParser(description='extract iris area from eye image')
parser.add_argument('source',nargs='?', help='directory of images', type=str, default='sgtuz')
parser.add_argument('outdir', type=str, nargs='?', help="out directory to save masked iris", default='det1280')
parser.add_argument('weights',nargs='?', help='weights path', type=str, default='models/iris8n_1280.pt')
parser.add_argument('imgsz', type=str, nargs='?', help="image size 1280", default=1280)
parser.add_argument('conf', type=str, nargs='?', help=" box threshold 0.25", default=0.8)
parser.add_argument('iris_id', type=str, nargs='?', help="iris class_id",default=0)
parser.add_argument('half', type=str, nargs='?', help="half=False fp16", default=False)
#parser.add_argument('device', type=str, nargs='?', help="cpu=>cpu, cuda=>0", default='cpu')
parser.add_argument('device', type=str, nargs='?', help="cpu=>cpu, cuda=>0", default=0)
args = parser.parse_args()
print("args:",args)
q =Detector(args)
q.crop()
