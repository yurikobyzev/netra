import cv2
from datetime import datetime
from typing import Tuple, List
from gradio_client.serializing import ImgSerializable
import gradio as gr
import os
import torch
import numpy as np
import pandas as pd
import random
from groundingdino.util.inference import load_model, load_image, predict #, annotate
import groundingdino.datasets.transforms as T
import supervision as sv
from torchvision.ops import box_convert
from PIL import Image
import ultralytics


grtransform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def grpredict(model_type,model,image,text_prompt,box_threshold,text_threshold):
    print("modtype==",model_type)
    (H,W,C) = np.array(image).shape
    print(H,W,C)
    if model_type=='gdino':
        im = Image.fromarray(image)
        image,_ = grtransform(im, None)
        print(image.size())
        classes = text_prompt.split('.')
        classes = [cl.strip() for cl in classes]
        class_id=[]
        boxes, logits, phrases = predict(
          model=model,
          image=image,
          device='cpu',
          caption=text_prompt,
          box_threshold=box_threshold,
          text_threshold=text_threshold,
          remove_combined=True,
        )
        for ph in phrases: 
            for i in range(len(classes)): 
                print('found:==================',ph)
                found=False

                if classes[i]==ph:
                    class_id.append(i)
                    found=True
                    break
            if not found:
                class_id.append(0)

        detections={'boxes': boxes.numpy(),'classes': classes,'class_id': np.array(class_id)}
    return detections




def EYEmodel(model_type='gdino'):
    model_type=model_type
    if model_type=='gdino':
        CONFIG_PATH = "GroundingDINO_SwinT_OGC.py"
        WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
        print(CONFIG_PATH, WEIGHTS_PATH)
        model = load_model(CONFIG_PATH, WEIGHTS_PATH)
        model.eval()
    return model



class EYE:
    def __init__(self,model_type='gdino'):
        self.headers = ['label','xc','yc','w','h','label','file_output']
        self.model = EYEmodel(model_type)
        self.model_type=model_type
        self.halfanglewidth = 3
        print('init model:',self.model_type)
        self.create_ui()



    def upload_file(self,files): 
        return files,gr.Image(files)

    def savepredict(self,pre_txt,save_filter,file_output,labels_table):
        splitarr=pre_txt.split('\n')
        print('SAVEFILTER:',save_filter)
        index = save_filter.split(',')
        for i in index:
            i = int(i)
            arr=[]
            cl_id,xc,yc,w,h,i,label = splitarr[i].split(',')
            arr = [cl_id,xc,yc,w,h,label]
            print(arr)
            if (file_output != None and file_output !=''):
                file_txt = os.path.basename(file_output)+'.txt'
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

    def filter(self,boxes,class_id,r):
        boxesf=[]
        class_f=[]
        for box,cl in zip(boxes,class_id): 
            xc,yc,w,h=box
            print(box)
            if ((h>r['rmin'] and h<r['rmax'] and w<=h) or (w>r['rmin'] and w<r['rmax'] and h<=w)) and (xc<r['maxx'] and xc>r['minx'] and yc<r['maxy'] and yc>r['miny']): 
                boxesf.append(list(box))
                class_f.append(cl)
        return np.array(class_f),np.array(boxesf)        

    def annotate(self,image_source,d,r,draw_noise):
        boxes = d['boxes']
        class_id = d['class_id']
        classes = d['classes']
        print('class_id:',class_id)
        print('classes',classes)
        if not draw_noise and len(class_id)>0:
            class_id,boxes = self.filter(boxes,class_id,r)
        h, w, _ = image_source.shape
        boxes =  torch.Tensor(boxes)
        boxeshw = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxeshw,in_fmt="cxcywh",out_fmt="xyxy").cpu().numpy()
        svdetections = sv.Detections(xyxy=xyxy,class_id=class_id)
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        annotated_frame = bounding_box_annotator.annotate(scene=image_source,detections=svdetections)

        stitch_start = 0
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
        print('r:',r)
        rrminx = r['minx']*w
        rrminy = r['miny']*h
        rrmaxx = r['maxx']*w
        rrmaxy = r['maxy']*h
        rrminr = r['rmin']*h/2
        rrmaxr = r['rmax']*h/2


        start_point = (int(rrminx),int(rrminy))
        end_point = (int(rrmaxx),int(rrmaxy))
        #annotated_frame = cv2.rectangle(annotated_frame, start_point, end_point, color, thickness) 
        print('1:',start_point,end_point)


        csv_result=[]
        savef = []
        i=-1
        for box,nbox in zip(xyxy,boxes.cpu().numpy().astype(float)): 
            xc, yc, w, h =  nbox 

            i+=1
            x1,y1,x2,y2 = box
            center_coordinates = (int(0.5*(x1+x2)), int(0.5*(y1+y2)))
            #annotated_frame = cv2.putText(annotated_frame, str(i), center_coordinates, font, fontScale, colorl, thickness, cv2.LINE_AA) 
            right_coordinates = (int(x2), int(y2))
            left_coordinates = (int(x1), int(y1))
            axesLength = (int(0.5*(x2-x1)), int(0.5*(y2-y1))) 
            saxes = str(w)+','+str(h)
            scent = str(xc)+','+str(yc)
            if ((h>r['rmin'] and h<r['rmax'] and w<=h) or (w>r['rmin'] and w<r['rmax'] and h<=w)) and (xc<r['maxx'] and xc>r['minx'] and yc<r['maxy'] and yc>r['miny']): 
                deltaxp = int(axesLength[0]*(1+1/7))
                deltaxm = int(axesLength[0]*(1-1/7))
                deltayp = int(axesLength[1]*(1+1/7))
                deltaym = int(axesLength[1]*(1-1/7))
                axesLengthp=(deltaxp,deltayp)
                axesLengthm=(deltaxm,deltaym)

                #annotated_frame = cv2.ellipse(annotated_frame, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 
#                for idx in range(self.stitches.value):
#                    startAngle=stitch_start+idx*360/self.stitches.value-self.halfanglewidth
#                    endAngle=startAngle+2*self.halfanglewidth
                    #annotated_frame = cv2.ellipse(annotated_frame, center_coordinates, axesLengthp, angle, startAngle, endAngle, color, thickness) 
                    #annotated_frame = cv2.ellipse(annotated_frame, center_coordinates, axesLengthm, angle, startAngle, endAngle, color, thickness) 

                annotated_frame = cv2.ellipse(annotated_frame, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 
                annotated_frame = cv2.putText(annotated_frame, str(i), right_coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
                savef.append(str(i)) 
            annotated_frame = cv2.putText(annotated_frame, str(i), left_coordinates, font, fontScale, colorl, thickness, cv2.LINE_AA) 
            csv_result.append(str(class_id[i])+','+scent+','+saxes+','+ str(i)+','+classes[class_id[i]]) 
        return (annotated_frame, '\n'.join(csv_result),','.join(savef))

    def gradio_predict(self,img,text_prompt,text_threshold,box_threshold,rmin_threshold,rmax_threshold,minx_threshold,maxx_threshold,miny_threshold,maxy_threshold,draw_noise,file_output,model_type="gdino"):
        restrictions = {'rmin': rmin_threshold/100.0,'rmax': rmax_threshold/100.0,'minx': minx_threshold/100.0,'maxx': maxx_threshold/100.0,'miny': miny_threshold/100.0,'maxy': maxy_threshold/100.0}
        detections = grpredict(self.model_type,self.model,img,text_prompt,text_threshold/100.0,box_threshold/100.0)
        annotated_frame,csv_result,save_filter = self.annotate(img,detections,restrictions,draw_noise)
        return (annotated_frame, csv_result, save_filter)

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
                text_prompt=gr.Textbox(interactive=True,value='blue yellow brown medium iris.small black oval pupil in iris')
                text_threshold = gr.Slider(0, 100, value=35, interactive=True, label="text_prompt thr", info="Choose between 0 and 100")
                box_threshold = gr.Slider(0, 100, value=10, interactive=True, label="box thr", info="Choose between 0 and 100")
              with gr.Column():
                rmin_threshold = gr.Slider(0, 100, value=7, interactive=True, label="min value for radius ", info="Choose between 0 and 100")
                rmax_threshold = gr.Slider(0, 100, value=45, interactive=True, label="max value for radius ", info="Choose between 0 and 100")
              with gr.Column():
                minx_threshold = gr.Slider(0, 100, value=20, interactive=True, label="min value for x_center ", info="Choose between 0 and 100")
                maxx_threshold = gr.Slider(0, 100, value=80, interactive=True, label="max value for x_center ", info="Choose between 0 and 100")
              with gr.Column():
                miny_threshold = gr.Slider(0, 100, value=20, interactive=True, label="min value for y_center ", info="Choose between 0 and 100")
                maxy_threshold = gr.Slider(0, 100, value=85, interactive=True, label="max value for y_center ", info="Choose between 0 and 100")
              with gr.Row(): 
                with gr.Column(): 
                    model_type = gr.Dropdown( 
                            choices = ["gdino"], 
                            value = "gdino",
                            )

                with gr.Column(): 
                    predict_btn = gr.Button("predict with new params/examples")
                    draw_noise = gr.Checkbox(label='С ложными предиктами',value=True)
            with gr.Row():
                with gr.Column():
                  img_output = gr.Image(height=800,width=1200)
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
            predict_btn.click(self.gradio_predict, [img_input,text_prompt,text_threshold,box_threshold,rmin_threshold,rmax_threshold,minx_threshold,maxx_threshold,miny_threshold,maxy_threshold,draw_noise,file_output,model_type], [img_output,pre_txt,save_filter])

            with gr.Row(): 
                gr.Examples( examples = [
                    ["sgtuz/0000000001_left_manual10-05-2023_13-43-42.png",
                    "blue yellow brown medium iris.small black oval pupil in iris",35,10,7,45,20,80,20,85,True,
                    "sgtuz/0000000001_left_manual10-05-2023_13-43-42.png","gdino"],

                    ["sgtuz/0000000001_right_manual10-05-2023_13-43-47.png",
                    "blue yellow brown medium iris.small black oval pupil in iris",35,10,7,45,20,80,20,85,True,
                    "sgtuz/0000000001_right_manual10-05-2023_13-43-47.png","gdino"],


                    ["sgtuz/0000000002_left_manual10-05-2023_13-46-08.png",
                    "blue yellow brown medium iris.small black oval pupil in iris",35,10,7,45,20,80,20,85,True,
                    "sgtuz/0000000002_left_manual10-05-2023_13-46-08.png","gdino"],

                    ["sgtuz/0000000002_right_manual10-05-2023_13-46-04.png",
                    "blue yellow brown medium iris.small black oval pupil in iris",35,10,7,45,20,80,20,85,True,
                    "sgtuz/0000000002_right_manual10-05-2023_13-46-04.png","gdino"],

                    ["sgtuz/0000000003_left_manual10-05-2023_13-48-07.png",
                    "blue yellow brown medium iris.small black oval pupil in iris",35,10,7,45,20,80,20,85,True,
                    "sgtuz/0000000003_left_manual10-05-2023_13-48-07.png","gdino"],

                    ["sgtuz/0000000003_right_manual10-05-2023_13-48-28.png",
                    "blue yellow brown medium iris.small black oval pupil in iris",35,10,7,45,20,80,20,85,True,
                    "sgtuz/0000000003_right_manual10-05-2023_13-48-28.png","gdino"],
                    ], 
                        inputs = [img_input,text_prompt,text_threshold,box_threshold,rmin_threshold,rmax_threshold,minx_threshold,maxx_threshold,miny_threshold,maxy_threshold,draw_noise,file_output], 
                        outputs = [img_output,pre_txt,save_filter],
                        fn = self.gradio_predict,
                        run_on_click = True,
                        cache_examples=True,
                        )



#                0000000001_left_manual10-05-2023_13-43-42.png   0000000018_right_manual10-05-2023_14-19-16.png
#                0000000001_right_manual10-05-2023_13-43-47.png  0000000019_left_manual10-05-2023_14-21-08.png
#                0000000002_left_manual10-05-2023_13-46-08.png   0000000019_right_manual10-05-2023_14-20-55.png
#                0000000002_right_manual10-05-2023_13-46-04.png  0000000020_left_manual10-05-2023_14-22-30.png
#                0000000003_left_manual10-05-2023_13-48-07.png   0000000020_right_manual10-05-2023_14-22-43.png
#                0000000003_right_manual10-05-2023_13-48-28.png  0000000021_left_manual10-05-2023_14-24-40.png
#                0000000004_left_manual10-05-2023_13-50-40.png   0000000021_right_manual10-05-2023_14-24-51.png
#                0000000004_right_manual10-05-2023_13-50-27.png  0000000022_left_manual10-05-2023_14-26-33.png
#                0000000005_left_manual10-05-2023_13-52-34.png   0000000022_right_manual10-05-2023_14-26-32.png
#                0000000005_right_manual10-05-2023_13-52-31.png  0000000023_left_manual10-05-2023_14-28-27.png
#                0000000006_left_manual10-05-2023_13-54-51.png   0000000023_right_manual10-05-2023_14-28-28.png
#                0000000006_right_manual10-05-2023_13-54-43.png  0000000024_left_manual10-05-2023_14-30-05.png
#                0000000007_left_manual10-05-2023_13-57-05.png   0000000024_right_manual10-05-2023_14-30-09.png
#                0000000007_right_manual10-05-2023_13-57-05.png  0000000025_left_manual10-05-2023_14-31-46.png
#                0000000008_left_manual10-05-2023_13-59-35.png   0000000025_right_manual10-05-2023_14-31-44.png
#                0000000008_right_manual10-05-2023_13-59-34.png  0000000026_left_manual10-05-2023_14-33-12.png
#                0000000009_left_manual10-05-2023_14-02-08.png   0000000026_right_manual10-05-2023_14-32-59.png
#                0000000009_right_manual10-05-2023_14-02-07.png  0000000027_left_manual10-05-2023_14-34-42.png
#                0000000010_left_manual10-05-2023_14-04-19.png   0000000027_right_manual10-05-2023_14-34-44.png
#                0000000010_right_manual10-05-2023_14-04-18.png  0000000028_left_manual10-05-2023_14-36-25.png
#                0000000011_left_manual10-05-2023_14-06-08.png   0000000028_right_manual10-05-2023_14-36-33.png
#                0000000011_right_manual10-05-2023_14-06-04.png  0000000029_left_manual10-05-2023_14-38-06.png
#                0000000012_left_manual10-05-2023_14-07-39.png   0000000029_right_manual10-05-2023_14-38-05.png
#                0000000012_right_manual10-05-2023_14-07-43.png  0000000030_left_manual10-05-2023_14-40-04.png
#                0000000013_left_manual10-05-2023_14-10-06.png   0000000030_right_manual10-05-2023_14-40-06.png
#                0000000013_right_manual10-05-2023_14-10-03.png  0000000031_left_manual10-05-2023_14-42-10.png
#                0000000014_left_manual10-05-2023_14-12-03.png   0000000031_right_manual10-05-2023_14-42-01.png
#                0000000014_right_manual10-05-2023_14-12-02.png  0000000032_left_manual10-05-2023_14-43-45.png
#                0000000015_left_manual10-05-2023_14-13-32.png   0000000032_right_manual10-05-2023_14-43-30.png
#                0000000015_right_manual10-05-2023_14-13-38.png  0000000033_left_manual10-05-2023_14-45-23.png
#                0000000016_left_manual10-05-2023_14-15-45.png   0000000033_right_manual10-05-2023_14-45-22.png
#                0000000016_right_manual10-05-2023_14-15-48.png  0000000034_left_manual10-05-2023_14-47-14.png
#                0000000017_left_manual10-05-2023_14-17-25.png   0000000034_right_manual10-05-2023_14-47-08.png
#                0000000017_right_manual10-05-2023_14-17-34.png  0000000035_left_manual10-05-2023_14-49-14.png
#                0000000018_left_manual10-05-2023_14-19-14.png   0000000035_right_manual10-05-2023_14-49-13.png

