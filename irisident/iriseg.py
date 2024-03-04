import cv2
import os
from segmentor.Segmentor import Segmentor
import argparse
import datetime

parser = argparse.ArgumentParser(description='extract iris area from eye image')
parser.add_argument('source',nargs='?', help='directory of images', type=str, default='sgtuz')
parser.add_argument('outdir', type=str, nargs='?', help="out directory to save masked iris", default='out')
parser.add_argument('weights',nargs='?', help='weights path', type=str, default='models/yolov8s-seg-ds.pt')
parser.add_argument('imgsz', type=str, nargs='?', help="image size 640", default=1280)
parser.add_argument('conf', type=str, nargs='?', help=" box threshold 0.25", default=0.9)
parser.add_argument('iris_id', type=str, nargs='?', help="iris class_id",default=2)
parser.add_argument('half', type=str, nargs='?', help="half=False fp16", default=False)
parser.add_argument('device', type=str, nargs='?', help="cpu=>cpu, cuda=>0", default=0)
args = parser.parse_args()
print("args:",args)
q =Segmentor(args)
q.crop()
