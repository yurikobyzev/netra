import cv2
import numpy as np
import math
from scipy.spatial import distance as dist
import classes_iris

#=============================================================================================#
# Обработка фото нейронной сетью
#=============================================================================================#

# подготовка фото для нейронной сети
def img_processing(img_orig):
    tmp_img = img_orig.copy()
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)
    tmp_img = cv2.equalizeHist(tmp_img)
    tmp_img = cv2.resize(tmp_img, (180,108))[np.newaxis, ..., np.newaxis]
    return tmp_img

# вычисление коффииентов масштабирования фото
def get_koeff(img):
    h, w = img.shape[:2]
    ky = w/180 # 1920
    kx = h/108 # 1080
    print((kx,ky))
    #print("y=",img.shape[0], " x=", img.shape[1])
    return (kx,ky)

# получение lms (все точки лимба, глазной щели и зрачка)
def get_lms(img_orig, model_limb):
    tmp_img = img_processing(img_orig)
    tmp_pred_img = model_limb.net.predict(tmp_img/255 * 2 - 1)
    tmp_lms_img = model_limb._calculate_landmarks(tmp_pred_img)
    return tmp_lms_img

#=============================================================================================#
# Лимб
#=============================================================================================#

# получение точек лимба из ключевых точек
def get_limb_points(lms_in, kx, ky):
    limb_points = []
    for i, lm in enumerate(np.squeeze(lms_in)):
        y, x = int(lm[0]*3*ky), int(lm[1]*3*kx)
        if ((i < 16) and (i >=8)):
            limb_points.append([y, x]) # точки лимба
    return limb_points

# получение центра лимба
def get_limb_center(limb_points):
    limb_center = np.zeros((2,))
    for (y, x) in limb_points:
        limb_center += (y,x)
    limb_center = (limb_center/8).astype(np.int32)
    return limb_center

# вычисление радиуса лимба
def get_limb_radius(limb_points):
    l_x = []
    l_y =[]
    for i in range(8):
        l_y.append(limb_points[i][1])
        l_x.append(limb_points[i][0])
    max_x, min_x, max_y, min_y = max(l_x), min(l_x), max(l_y), min(l_y)
    r = int(math.trunc(min(max_x-min_x, max_y-min_y) / 2))
    return r

# получение ROI лимба
def get_limb_ROI(limb_points):
    R = get_limb_radius(limb_points)
    C = get_limb_center(limb_points)
    limb_ROI = (C[1]-R,C[1]+R,C[0]-R,C[0]+R)
    return limb_ROI

# получение лимба
def get_limb(frame, model_limb):
    kx, ky = get_koeff(frame)
    limb_points = get_limb_points(get_lms(frame, model_limb), kx, ky)
    center = get_limb_center(limb_points)
    center_pupil = (0,0)
    radius = get_limb_radius(limb_points)
    roi = get_limb_ROI(limb_points)
    LIMB = classes_iris.tlimb(center,center_pupil,radius,roi, limb_points)
    return LIMB

#=============================================================================================#
# Глазная щель
#=============================================================================================#

# получение точек глазной щели из ключевых точек
def get_eyelid_points(lms_in, kx, ky):
    try:
        eyelids_points = []
        limb_points = []
        for i, lm in enumerate(np.squeeze(lms_in)):
            y, x = int(lm[0]*3*ky), int(lm[1]*3*kx)
            if i < 8:
                eyelids_points.append([y, x]) # точки глазной щели
            if ((i < 16) and (i >=8)):
                limb_points.append([y, x]) # точки лимба
            if i == 16:
                eyelids_points.append([y, x]) # предсказанный центр зрачка
            if i == 17:
                eyelids_points.append([y, x]) # центр глазного яблока
        return (eyelids_points, limb_points)
    except Exception as err:
        print("[ERROR]: Ошибка получения точек глазной щели из ключевых точек: ", str(err))
        return False

# получение ROI глазной щели
def get_eyelid_ROI(eyelid_points):
    try:
        l_x = []
        l_y =[]
        for i in range(8):
            l_y.append(eyelid_points[i][0])
            l_x.append(eyelid_points[i][1])
        max_x, min_x, max_y, min_y = max(l_x), min(l_x), max(l_y), min(l_y)
        #r = int(min(max_x-min_x,max_y-min_y) / 2)
        eyelid_ROI = (min_y,max_y,min_x,max_x)
        #eyelid_ROI = (min_x,max_x,min_y,max_y)
        return eyelid_ROI
    except Exception as err:
        print("[ERROR]: Ошибка получения ROI глазной щели: ", str(err))
        return False

# получение глазной щели
def get_eyelid(frame, model_limb):
    try:
        kx, ky = get_koeff(frame)
        eyelid_points, limb_points = get_eyelid_points(get_lms(frame, model_limb),kx, ky)
        roi = get_eyelid_ROI(eyelid_points)
        r = get_limb_radius(limb_points)
        center = get_limb_center(limb_points)
        pupil_center = eyelid_points[8]
        eyelid_center = eyelid_points[9]
        EYELID = classes_iris.teyelid(center, r, roi, pupil_center, eyelid_center, eyelid_points, limb_points)
        return EYELID
    except Exception as err:
        print("[ERROR]: Ошибка получения глазной щели: ", str(err))
        return False