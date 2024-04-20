from pathlib import Path
import time
from PIL import Image
from pydantic import BaseModel
import numpy as np
import os
import glob
import numpy as np
import lancedb
import pyarrow as pa
from oml.models import ViTExtractor
from oml.inference.flat import inference_on_images
from oml.registry.transforms import get_transforms_for_pretrained

uri = "data/vits16"
db = lancedb.connect(uri)

schema = pa.schema([pa.field("embeddings", pa.list_(pa.float32(), list_size=384)),
         pa.field("label", pa.string()),
         pa.field("filename", pa.string()),
    ])

arch='vits16'

weights='models/iris_vits16_cpu.pt'
modeltype='vits16_dino'
tbl = db.open_table(modeltype)
imgsz=384
imgdir = 'outm1280'
model=ViTExtractor(weights=weights,arch=arch, normalise_features=False)
model.eval()
transform, imread_pillow = get_transforms_for_pretrained(modeltype)
args = {"num_workers": 0, "batch_size": 8}
imgpath=glob.glob(imgdir+'/*.png')

for f in imgpath:
    basename=Path(f).stem
    ip = imread_pillow(f)
    print(type(ip))
    print(basename)
    tens = transform(ip)
    start=time.time()
    features = model(tens.unsqueeze(0))
    stop=time.time()
    print('inference:',stop-start)
    #        features = inference_on_images(model, paths=imgpath, transform=transform, **args)
    feat = features[0].detach().cpu().numpy().tolist()
    res = tbl.search(feat).limit(3).to_pandas()
    stop=time.time()
    print('inference+search:',stop-start)
    res.drop(['embeddings'], axis=1, inplace=True)   
    res.to_csv(imgdir+'/'+basename+'.csv')
