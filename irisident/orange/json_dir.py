import os
import json
from pathlib import Path
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
import pyarrow as pa
import lancedb

uri = "data/vits16"
db = lancedb.connect(uri)
schema = pa.schema([pa.field("embeddings", pa.list_(pa.float32(), list_size=384)),
         pa.field("label", pa.string()),
         pa.field("filename", pa.string()),
    ])


arch='vits16'
weights='models/iris_vits16_cpu.pt'
modeltype='vits16_dino'
tbl = db.create_table(modeltype, schema=schema)
imgsz=384



zpath='result'
embedding=zpath+'_'+modeltype+'.json'





def walk_file_or_dir(root):
    if os.path.isfile(root):
        dirname, basename = os.path.split(root)
        yield dirname, [], [basename]
    else:
        for path, dirnames, filenames in os.walk(root):
            yield path, dirnames, filenames

def main():                    
    model=ViTExtractor(weights=weights,arch=arch, normalise_features=False)
#    model.to('cuda')
    model.eval()
    transform, im_reader = get_transforms_for_pretrained(modeltype)
    args = {"num_workers": 0, "batch_size": 8}
    imgpath=[]
    labels=[]
    for path, dirnames, filenames in walk_file_or_dir(zpath):
            for f in filenames: 
                if (f.endswith('.png') or f.endswith('.jpg')):
                    fp=os.path.join(path,f)
                    imgpath.append(fp)
                    label_=Path(f).stem
                    label=label_.split('_',maxsplit=1)[0]
                    labels.append(label)


    start=time.time()
    features_queries = inference_on_images(model, paths=imgpath, transform=transform, **args)
    stop=time.time()
    print(stop-start)
    result=[]
    for (fe,path,lab) in zip(features_queries,imgpath,labels):
        ee = ','.join(map(str,fe.numpy().tolist()))
        result.append({'label':lab,'filename': path,'embeddings': fe.cpu().numpy().tolist()})
    r=json.dumps(result)


    with open(embedding,'w') as jtxt:
        jtxt.write(r)
        tbl = db.create_table(modeltype,data=result,schema=schema)
    exit(0)

if __name__ == "__main__":
    main()



