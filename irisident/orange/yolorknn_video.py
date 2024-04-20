import cv2
from PIL import Image
import numpy as np
from py_utils.coco_utils import COCO_test_helper
import os
import time
import sys
import requests

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
#sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))



OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

#IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)
IMG_SIZE = (1280, 1280)  # (width, height), such as (1280, 736)

CLASSES=["iris","pupil"]
########################

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def softmax(x, axis=None):
    x = x - x.max(axis = axis, keepdims=True)
    y = np.exp(x)
    return y/y.sum(axis = axis, keepdims=True)


def dfl(position):
    # Distribution Focal Loss (DFL)
    n,c,h,w = position.shape
    p_num = 4
    mc = c//p_num
    y = position.reshape(n,p_num,mc,h,w)
    y = softmax(y,2)
    acc_metrix = np.array(range(mc),dtype = float).reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y




#def dfl(position):
#    # Distribution Focal Loss (DFL)
#    import torch
#    x = torch.tensor(position)
#    n,c,h,w = x.shape
#    p_num = 4
#    mc = c//p_num
#    y = x.reshape(n,p_num,mc,h,w)
#    y = y.softmax(2)
#    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
#    y = (y*acc_metrix).sum(2)
#    return y.numpy()


def box_process(position):
    print("POSITION:",position.shape)
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


######################
# embedding label predictor URL
url = "http://127.0.0.1:8080/imagesearch"
url = "http://10.10.0.1:8080/imagesearch"

# Load the YOLOv8 model

model_path = 'models/iris8n_1280.rknn'
platform = 'rknn'
target = 'RK3588'
device_id=None

from py_utils.rknn_executor import RKNN_model_container
model = RKNN_model_container(model_path, target, device_id)
co_helper = COCO_test_helper(enable_letter_box=True)


# Open the video file
video_path = "/dev/video1"
cap = cv2.VideoCapture(video_path)
font = cv2.FONT_HERSHEY_DUPLEX

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
    # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        pad_color = (0,0,0)
        img = co_helper.letter_box(im= frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        input_data = img
        start=time.time()
        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs)
        end = time.time()
        print("inference time with pp:",end - start)
        annotated_frame = frame.copy()
        if boxes is not None:
            for box, score, cl in zip(co_helper.get_real_box(boxes), scores, classes):
                top, left, right, bottom = [int(_b) for _b in box]
                print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
#                annotated_frame = cv2.rectangle(annotated_frame, (top, left), (right, bottom), (255, 0, 0), 2)
#                annotated_frame = cv2.putText(annotated_frame, '{0} {1:.2f}'.format(CLASSES[cl], score),
#                            (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if cl==0: # iris
                    image_pil=Image.fromarray(frame,'RGB')
                    ff = image_pil.crop((top,left,right,bottom))
                    array_list = np.array(ff).tolist()
                    start=time.time()
                    response = requests.post(url, json={"array": array_list})
                    stop=time.time()
                    elapsed=str(stop-start)
                    if response.status_code == 200:
                        import json
                        rs = response.json()
                        print(type(rs), rs)
                        s = str(rs["label"])+' '+str(rs["_distance"]+' delay: '+elapsed)
                        annotated_frame = cv2.rectangle(annotated_frame, (top, left), (right, bottom), (0, 255, 0), 1)
                        annotated_frame = cv2.putText(annotated_frame, s, (top + 6, left - 6), font, 1.0, (255, 255, 255), 1)
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
