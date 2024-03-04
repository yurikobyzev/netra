from segmentation import Segmentator
from elg_keras import KerasELG # Модель детекции ключевых точек глаза
import core_iris
import os
import numpy as np
import cv2
import math

# Создаем модель сегментации
def get_model_seg(path):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1" # отключаем использование графического ядра
        print("Создаем модель сегментации...")
        _model = KerasELG()
        print("Загружаем веса модели сегментации...")
        _model.net.load_weights(path)# Загружаем веса нейронной сети
        _frame = np.zeros((100,100,3), np.uint8)
        _frame[:,:] = (193,186,71)
        _limb = core_iris.get_limb(_frame, _model)
        return _model
    except Exception as err:
        print("Ошибка создания модели сегментации: " + path + ": " + str(err))

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

if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.abspath(__file__))  # директория запуска риложения
    model = get_model_seg(os.path.join(base_folder, 'elg_keras.h5'))
    SEGMENTATOR = Segmentator(model)
    img_path = os.path.join(base_folder, 'img_orig', '0000000001_left_manual10-05-2023_13-43-31.png')
    img = cv2.imread(img_path)
    #_img = img.copy()
    #_img = SEGMENTATOR.draw_iris_poligon(_img) # нарисовать радужку и глазную щель
    #cv2.imshow('image_seg', _img)
    #cv2.waitKey(0)
    _img_cropped = SEGMENTATOR.cropped_iris(SEGMENTATOR.segmented_iris_polygon(img))
    cv2.imshow('image_crop', _img_cropped)
    cv2.waitKey(0)
    _img_rectangles = _img_cropped.copy()
    coords = get_rect(_img_rectangles)
    cv2.rectangle(_img_rectangles, coords[0], coords[1], (0, 255, 255), 1)
    cv2.rectangle(_img_rectangles, coords[2], coords[3], (0, 255, 255), 1)
    cv2.imshow('image_rect', _img_rectangles)
    cv2.waitKey(0)
    _img_rect_l = _img_cropped[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
    _img_rect_r = _img_cropped[coords[2][1]:coords[3][1], coords[2][0]:coords[3][0]]
    cv2.imshow('image_rect_left', _img_rect_l)
    cv2.waitKey(0)
    cv2.imshow('image_rect_right', _img_rect_r)
    cv2.waitKey(0)
    cv2.destroyAllwindows()


