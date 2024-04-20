import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests

# Load the YOLOv8 model
# model = YOLO('models/iris8n_1280.onnx')
model = YOLO('models/iris8n_1280.pt')
url = "http://10.10.0.1:8080/imagesearch"

# Open the video file
video_path = "/dev/video1"
cap = cv2.VideoCapture(video_path)
font = cv2.FONT_HERSHEY_DUPLEX

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        #results = model.predict(frame,imgsz=1280,classes=0,conf=0.8,stream=True)
        results = model.predict(frame,imgsz=1280,classes=0)
        #r = results[0]
        for r in results:
            # Visualize the results on the frame
            annotated_frame = r.plot()
            if r.boxes.shape[0]>0:
                boxes = r.boxes.cpu().numpy()
                box = boxes.xyxy
                if len(box)>0:
                    box = box[0]
                    print('detected:',box)
                    x1, y1, x2, y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                    image_pil=Image.fromarray(r.orig_img,'RGB')
                    ff = image_pil.crop((x1,y1,x2,y2))
                    array_list = np.array(ff).tolist()
                    response = requests.post(url, json={"array": array_list})
                    if response.status_code == 200:
                        import json
                        rs = response.json()
                        print(type(rs), rs)
                        s = str(rs["label"])+' '+str(rs["_distance"])
                        annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        annotated_frame = cv2.putText(annotated_frame, s, (x1 + 6, y1 - 6), font, 1.0, (255, 255, 255), 1)
                    else:
                        print("Ошибка при выполнении запроса:", response.status_code)

            # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
