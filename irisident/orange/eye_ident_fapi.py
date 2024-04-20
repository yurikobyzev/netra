from fastapi import FastAPI
import torch
import uvicorn
from PIL import Image
from pydantic import BaseModel
import numpy as np
import os
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

limit = 1 # return limit nearest

app = FastAPI()
arch='vits16'
#weights='models/iris_vits16_gpu.pt'
weights='models/iris_vits16_cpu.pt'
modeltype='vits16_dino'
tbl = db.open_table(modeltype)
imgsz=384
model=ViTExtractor(weights=weights,arch=arch, normalise_features=False)
model.eval()
#model = model.to('cuda')
transform, imread_pillow = get_transforms_for_pretrained(modeltype)
args = {"num_workers": 0, "batch_size": 8}
limit=1


class NumpyArray(BaseModel):
    array: list

@app.post("/imagesearch")
def imagesearch(numpy_array: NumpyArray):
    print('got array:')
   #print(numpy_array.array)

    if len(numpy_array.array)==0:
        print('error empty request')
        return {"label": '',"filename": '',"_distance": -1.0}
    arr = np.array(numpy_array.array,dtype=np.uint8)
    print('shape:',arr.shape)
    try:
        tens = Image.fromarray(arr,'RGB')
        print('get array:',type(tens))
        print(transform)
        tens = transform(tens)
#        tens = tens.to('cuda')
        print('tens:')
        features = model(tens.unsqueeze(0))
        print('features:',features)
        feat = features[0].detach().cpu().numpy().tolist()
        print('feat:',feat)
        df = tbl.search(feat).limit(1).to_pandas()
        lab = df['label'].values[0]
        dist=df['_distance'].values[0]

        res = {"label": str(lab), "_distance": str(dist)}

        print('res:',res)
        return res
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return {"label": 'Unknown',"filename": 'exception',"_distance": -1.0}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

