import cv2
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple, List
from gradio_client.serializing import ImgSerializable
import gradio as gr
import os
import torch
import numpy as np
import pandas as pd
import random
from ultralytics import YOLO
import supervision as sv
from torchvision.ops import box_convert
from PIL import Image
import ultralytics
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



def predict(model,image,box_threshold):
    (H,W,C) = np.array(image).shape
    print(H,W,C)
    results=model.predict(image,conf=box_threshold)
    r = results[0]
    classes=model.names
    class_id = [int(r.boxes.cls[idx].item()) for idx in range(len(r.boxes))]
    conf = [(r.boxes.conf[idx].item()) for idx in range(len(r.boxes))]
    detections={'boxes': r.boxes.xywhn.cpu().numpy(),'classes': classes,'class_id': np.array(class_id),'conf': np.array(conf),'source_file': r.path}
    print("yolo detect")
    print(detections)
    return detections

def EYEmodel(model_type):
    model = YOLO(model_type)
    return model

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




def rectcrops(im_array,box):
    im = Image.fromarray(im_array[..., ::-1])
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
    return isolated, rectl_crop, rectr_crop



class EYE:
    def __init__(self,model_type='models/iris8n_1280.pt'):
        self.outdir = 'webimages'
        self.headers = ['label','xc','yc','w','h','label','conf','labelfile']
        self.sheaders = ['label','filename','_distance']
        self.model = EYEmodel(model_type)
        self.model_type=model_type
        self.arch='vits16'
        self.modeltype='vits16_dino'
        self.imgsz=384
        self.weights='models/iris_vits16_gpu.pt'
        self.idmodel=ViTExtractor(weights=self.weights,arch=self.arch, normalise_features=False)
        self.id_model = self.idmodel.to('cuda')
        self.idmodel.eval()
        self.uri = "data/vits16"
        self.db = lancedb.connect(self.uri)
        self.schema = pa.schema([pa.field("embeddings", pa.list_(pa.float32(), list_size=384)),
         pa.field("label", pa.string()),
         pa.field("filename", pa.string()),
        ])
        self.tbl = self.db.open_table(self.modeltype)
        #self.tbl.create_fts_index('label','filename')

        self.transform, self.imread_pillow = get_transforms_for_pretrained(self.modeltype)
        self.args = {"num_workers": 0, "batch_size": 8}
        self.create_ui()

    def emb_search(self,iso_emb,lbox_emb,rbox_emb,limit):
        result=[]
        for feat in [iso_emb,lbox_emb,rbox_emb]:
            #print(feat)
            feat = np.array([float(i) for i in feat.split(',')])
            res = self.tbl.search(feat).limit(limit).to_pandas()
            res.drop(['embeddings'], axis=1, inplace=True)
            result.append(res)
        return result



    def upload_file(self,files): 
        return files,gr.Image(files)

    def savepredict(self,pre_txt,save_filter,file_output,labels_table):
        splitarr=pre_txt.split('\n')
        print('SAVEFILTER:',save_filter)
        index = save_filter.split(',')
        for i in index:
            i = int(i)
            arr=[]
            cl_id,xc,yc,w,h,i,label,conf = splitarr[i].split(',')
            arr = [cl_id,xc,yc,w,h,label,conf]
            print(arr)
            if (file_output != None and file_output !=''):
                file_txt = Path(file_output).stem + '.txt'
            else:
                file_txt='labels.txt'
            if arr!=None: 
                arr.append(file_txt)
                labels_table.loc[len(labels_table)] = arr
        return labels_table

    def export_csv(self,df):
        now = datetime.now()
        fn="output_"+str(now)+".txt"
        df.to_csv(fn,index=False)
        return gr.File(value=fn, visible=True)

    def annotate(self,image_source,d):
        im = image_source.copy()
        boxes = d['boxes']
        class_id = d['class_id']
        classes = d['classes']
        conf = d['conf']
        h, w, _ = image_source.shape
        boxes =  torch.Tensor(boxes)
        boxeshw = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxeshw,in_fmt="cxcywh",out_fmt="xyxy").cpu().numpy()
        svdetections = sv.Detections(xyxy=xyxy,class_id=class_id)
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = bounding_box_annotator.annotate(scene=im,detections=svdetections)
        angle = 0
        startAngle = 0
        endAngle = 360
        color = (255, 255, 0) 
        colorr = (255,  0, 255) 
        colorl = (0, 255, 255) 
        thickness =3 
        thickness1 =1 
        fontScale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        csv_result=[]
        savef = []
        i=-1
        for box,nbox in zip(xyxy,boxes.cpu().numpy().astype(float)): 
            xc, yc, w, h =  nbox 
            i+=1
            x1,y1,x2,y2 = box
            center_coordinates = (int(0.5*(x1+x2)), int(0.5*(y1+y2)))
            right_coordinates = (int(x2), int(y2))
            left_coordinates = (int(x1), int(y1))
            axesLength = (int(0.5*(x2-x1)), int(0.5*(y2-y1))) 
            saxes = str(w)+','+str(h)
            scent = str(xc)+','+str(yc)
            annotated_frame = cv2.ellipse(annotated_frame, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 
            annotated_frame = cv2.putText(annotated_frame, str(i), right_coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            savef.append(str(i)) 
            annotated_frame = cv2.putText(annotated_frame, str(i), left_coordinates, font, fontScale, colorl, thickness, cv2.LINE_AA) 
            csv_result.append(str(class_id[i])+','+scent+','+saxes+','+ str(i)+','+classes[class_id[i]]+','+str(conf[i])) 
            if class_id[i]==0:
                iso, lbox, rbox = rectcrops(image_source,box)
                img_name = Path(d['source_file']).stem
                #filename = img_name+'.png'
                filename_iso = img_name+'_iso.png'
                filename_l = img_name+'_l.png'
                filename_r = img_name+'_r.png'
                #filename = os.path.join(self.outdir,filename)
                filename_l = os.path.join(self.outdir,filename_l)
                filename_r = os.path.join(self.outdir,filename_r)
                filename_iso = os.path.join(self.outdir,filename_iso)
                #imcrop.save(filename, "png")
                #cv2.imwrite(filename, iso_crop)
                pil_iso = Image.fromarray(iso)
                pil_iso.save(filename_iso)
                pil_lbox=Image.fromarray(lbox)
                pil_lbox.save(filename_l)
                pil_rbox=Image.fromarray(rbox)
                pil_rbox.save(filename_r)
                f_iso,f_lbox,f_rbox = inference_on_images(self.idmodel, paths=[filename_iso,filename_l,filename_r], transform=self.transform, **self.args)
                f_iso = f_iso.tolist()
                f_iso =  ",".join(str(element) for element in f_iso)
                f_lbox = f_lbox.tolist()
                f_lbox =  ",".join(str(element) for element in f_lbox)
                f_rbox = f_rbox.tolist()
                f_rbox =  ",".join(str(element) for element in f_rbox)
        return (annotated_frame, '\n'.join(csv_result),','.join(savef),iso,lbox,rbox,f_iso,f_lbox,f_rbox)

    def gradio_predict(self,img,box_threshold,file_output,model_type='models/iris8n_1280.pt'):
        print('model_type:',model_type)
        if self.model_type != model_type:
            print('model changed to:',model_type)
            self.model = EYEmodel(model_type)
            self.model_type = model_type

        detections = predict(self.model,img,box_threshold/100.0)
        annotated_frame,csv_result,save_filter,iso,lbox,rbox,f_iso,f_lbox,f_rbox  = self.annotate(img,detections)
        return (annotated_frame, csv_result, save_filter, iso,lbox,rbox,f_iso,f_lbox,f_rbox)

    def create_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("""
            **Grounding dino (project EYEris)) ** Kobyzev Yuri.
            """)
            with gr.Row():
                with gr.Column(): 
                    upload_button = gr.UploadButton("Click to Upload a File and than predict", file_types=["image"], file_count="single")
                    file_output = gr.File(visible=False)
                    img_input = gr.Image(visible=True,label='eye image',height=800,width=1200)
                    upload_button.upload(self.upload_file, upload_button, [file_output,img_input])
            with gr.Row():
              with gr.Column():
                box_threshold = gr.Slider(0, 100, value=75, interactive=True, label="box thr", info="Choose between 0 and 100")
              with gr.Row(): 
                with gr.Column(): 
                    model_type = gr.Dropdown( 
                            choices = ["models/iris8n_1280.pt","models/iris8n_640.pt"], 
                            value = "models/iris8n_1280.pt",
                            )

                with gr.Column(): 
                    predict_btn = gr.Button("predict with new params/examples")
            with gr.Row():
                with gr.Column():
                  img_output = gr.Image(height=800,width=1200)
            with gr.Row():
                with gr.Column():
                  iso = gr.Image(label='iris mask',height=200,width=200)
                  iso_emb = gr.Textbox(label='embedding masked iris',visible=False) 
                  iso_table=gr.Dataframe(
                            headers = self.sheaders,
                            interactive=False,
                            visible = True
                            )
                with gr.Column():
                  lbox = gr.Image(label='lbox',height=200,width=200)
                  lbox_emb = gr.Textbox(label='embedding lbox iris',visible=False) 
                  lbox_table=gr.Dataframe(
                            headers = self.sheaders,
                            interactive=False,
                            visible = True
                            )
                with gr.Column():
                  rbox = gr.Image(label='rbox',height=200,width=200)
                  rbox_emb = gr.Textbox(label='embedding rbox iris',visible=False) 
                  rbox_table=gr.Dataframe(
                            headers = self.sheaders,
                            interactive=False,
                            visible = True
                            )
            with gr.Row():
                    limit = gr.Slider(1, 10, value=3, interactive=True, label="число бпохожих", info="Choose between 1 and 10")
                    search_btn = gr.Button("Поиск по эмбеддингу в базе (после предикта радужки)")

            with gr.Row():
                    save_filter = gr.Textbox(label='save detected number') 
                    pre_txt = gr.Textbox()
            with gr.Row():
                with gr.Column():
                    labels_table=gr.Dataframe(
                            headers = self.headers,
                            interactive=False,
                            visible = True
                            )
                    savepredict_btn = gr.Button("save predict")
                    csv_btn = gr.Button(value="Выгрузить в csv file")
                    csv = gr.File(interactive=False, visible=False)
                    csv_btn.click(self.export_csv, labels_table, csv)
            savepredict_btn.click(self.savepredict, [pre_txt,save_filter,file_output,labels_table], [labels_table])
            search_btn.click(self.emb_search, [iso_emb,lbox_emb,rbox_emb,limit], [iso_table,lbox_table,rbox_table])
            predict_btn.click(self.gradio_predict, [img_input,box_threshold,file_output,model_type], [img_output,pre_txt,save_filter,iso,lbox,rbox,iso_emb,lbox_emb,rbox_emb])

            with gr.Row(): 
                gr.Examples( examples = [
                    ["sgtuz/0000000001_left_manual10-05-2023_13-43-42.png",90,
                    "sgtuz/0000000001_left_manual10-05-2023_13-43-42.png"],

                    ["sgtuz/0000000001_right_manual10-05-2023_13-43-47.png",90,
                    "sgtuz/0000000001_right_manual10-05-2023_13-43-47.png"],


                    ["sgtuz/0000000002_left_manual10-05-2023_13-46-08.png",90,
                    "sgtuz/0000000002_left_manual10-05-2023_13-46-08.png"],

                    ["sgtuz/0000000002_right_manual10-05-2023_13-46-04.png",90,
                    "sgtuz/0000000002_right_manual10-05-2023_13-46-04.png"],

                    ["sgtuz/0000000003_left_manual10-05-2023_13-48-07.png",90,
                    "sgtuz/0000000003_left_manual10-05-2023_13-48-07.png"],

                    ["sgtuz/0000000003_right_manual10-05-2023_13-48-28.png",90,
                    "sgtuz/0000000003_right_manual10-05-2023_13-48-28.png"],
                    ], 
                        inputs = [img_input,box_threshold,file_output], 
                        outputs = [img_output,pre_txt,save_filter,iso,lbox,rbox,iso_emb,lbox_emb,rbox_emb],
                        fn = self.gradio_predict,
                        run_on_click = True,
                        cache_examples=False,
                        )


