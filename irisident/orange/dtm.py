import tkinter as tk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageTk
import requests
import numpy as np

class App:
    window_size = (1280, 640)
    video_size = (1280, 640)
    url = "http://10.10.0.1:8080/imagesearch"



    def __init__(self, master):
        self.cam = [1,1]
        self.update_idx = 0
        self.names = ['unknown','unknown']
        self.update_period = 2
        self.model = YOLO('models/iris8n_1280.pt')
        self.master = master
        self.master.title("Приложение с видео 2 cameras")
        self.master.geometry(f"{self.__class__.window_size[0]}x{self.window_size[1]}")

        x = self.__class__.window_size[0]//2 - self.__class__.video_size[0]//2
        self.video_frame = tk.Frame(self.master)
        self.video_frame.place(x=x, y=150, width=self.__class__.video_size[0], height=self.__class__.video_size[1])

        self.video_box = tk.Label(self.video_frame)
        self.video_box.place(width=self.__class__.video_size[0], height=self.__class__.video_size[1])
        self.video_box.configure(bg="green")  # Задает цвет фона

        self.start_button = tk.Button(self.master, text="Запустить поток с камеры", command=self.start_stream)
        self.start_button.pack()

        self.status_bar = tk.Label(self.master, text="")
        self.status_bar.place(y=self.__class__.window_size[1]-20)


    def start_stream(self):
        self.caps =[]
        for i in self.cam:
            cap = cv2.VideoCapture(i)
            self.caps.append(cap)  # Номер устройства, если их несколько
        self.show_frame()
        self.start_button.config(state=tk.DISABLED)

    def show_frame(self):
        frames = []
        rets=[]
        finaly_frame = np.zeros((480,1280, 3), dtype=np.uint8)
        for i in self.cam:
            print('video:',i)
            ret, frame = self.caps[i-1].read()
            rets.append(ret)
            if ret:
                print('f read:',frame.shape)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #frame = cv2.resize(frame, (self.__class__.video_size[0]//2, self.__class__.video_size[1]//2))
            else:
                frame =  np.zeros((480,640,3), dtype=np.uint8)
            frames.append(frame)

        frame_index=-1
        for ret,frame in zip(rets,frames):        
            frame_index+=1
            if ret:
                results = self.model.predict(frame,classes=0)
                for i, r in enumerate(results[0]):
                    boxes = r.boxes.cpu().numpy()
                    box = boxes.xyxy
                    if len(box)>0:
                        box = box[0]
                        print('detected:',i,box)
                        x1, y1, x2, y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
#                        if self.update_idx % self.update_period == 0:
                        ff = frame[frame_index][x1:x2, y1:y2]
                        print('send:shape, x1,x2,y1,y2:',ff.shape,x1,x2,y1,y2)
                        array_list = ff
                        
                        response = requests.post(self.__class__.url, json={"array": array_list})
                        if response.status_code == 200:
                            import json
                            rs = response.json()
                            print(type(rs), rs)
                            s = str(rs["label"])+' '+str(rs["_distance"])
                            print("s=",s)
                            self.names[frame_index] = s
                        else:
                            print("Ошибка при выполнении запроса:", response.status_code)
                    cv2.rectangle(frames[frame_index], (x1, y1), (x2, y2), (0, 255, 0), 1)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frames[frame_index], self.names[frame_index], (x1 + 6, y1 - 6), font, 1.0, (255, 255, 255), 1)

                print('f0 shape',frames[0].shape)
                finaly_frame[:,:640] = frames[0]
                finaly_frame[:,640:] = frames[1]

                img = Image.fromarray(finaly_frame)
                img = ImageTk.PhotoImage(image=img)

                self.video_box.configure(image=img)
                self.video_box.image = img

                self.master.after(1, self.show_frame)
#                self.update_idx += 1

root = tk.Tk()
app = App(root)
root.mainloop()
