import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from oml.models import ViTExtractor
from oml.inference.flat import inference_on_images
from oml.registry.transforms import get_transforms_for_pretrained
from oml.utils.misc_torch import pairwise_dist
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.inference.flat import inference_on_dataframe
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from pprint import pprint


modeltype='vits16_dino'
arch='vits16'
weights='iris_dataset_LR1cpu.pt'
imgsz=384
zpath='ZENICA_IRIS'
embedding='zenica_iris_embedding.csv'






def walk_file_or_dir(root):
    if os.path.isfile(root):
        dirname, basename = os.path.split(root)
        yield dirname, [], [basename]
    else:
        for path, dirnames, filenames in os.walk(root):
            yield path, dirnames, filenames

def main():                    
    model=ViTExtractor(weights=weights,arch=arch, normalise_features=False)
    model.eval()
    transform = get_normalisation_resize_torch(im_size=imgsz)
    args = {"num_workers": 0, "batch_size": 8}
    imgpath=[]
    labels=[]
    for path, dirnames, filenames in walk_file_or_dir(zpath):
            for f in filenames: 
                if (f.endswith('.png')):
                    fp=os.path.join(path,f)
                    imgpath.append(fp)
                    lab = f.split('_')
                    label=str(int(lab[0]))
                    labels.append(label)


    print(len(labels))
    print(len(imgpath))

    features_queries = inference_on_images(model, paths=imgpath, transform=transform, **args)
    with open(embedding,'w') as f:
        for (fe,path,lab) in zip(features_queries,imgpath,labels):
            ee = ','.join(map(str,fe.numpy().tolist()))
            line = lab+'\t'+path+'\t'+ee+'\n'
            f.write(line)
    exit(0)

if __name__ == "__main__":
    main()



