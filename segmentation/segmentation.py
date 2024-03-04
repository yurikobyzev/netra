import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
import copy
import imutils
import core_iris
from scipy.interpolate import interp1d


class Segmentator():

    def __init__(self, model):
        self.model = model
        self.frame_right = np.zeros((480, 640, 3), np.uint8)
        self.frame_right[:, :] = (193, 186, 71)
        self.iris = None

    # функция для отрисовки лимба
    def draw_limb(self, frame):
        draw = frame.copy()
        y, x, r = self.iris.center[1]+10, self.iris.center[0]+3, self.iris.radius
        l = 10
        cv2.line(draw, (x - l, y), (x + l, y), (0, 0, 255), 4)  # отрисовка горизонтальной линии центра лимба
        cv2.line(draw, (x, y - l), (x, y + l), (0, 0, 255), 4)  # отрисовка вертикальной линии центра лимба
        cv2.circle(draw, (x, y), r, (0, 255, 255), 4)  # отрисовка окружности лимба
        return draw

    def draw_iris_poligon(self, img):
        #_iris = self.get_iris(img)
        _eye = self.get_eye(img)
        points_iris = _eye.limb_points
        points_iris = self.interpolation(points_iris + [points_iris[0]], method='cubic')
        points_eyelid = _eye.eyelid_points
        points_eyelid = self.interpolation(points_eyelid[:-2] + [points_eyelid[0]], method='cubic')
        cv2.polylines(img, pts=[np.array(points_iris, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.polylines(img, pts=[np.array(points_eyelid, np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
        return img

    def get_iris(self, foto):
        _limb = core_iris.get_limb(foto, self.model)
        return _limb

    def get_eye(self, foto):
        _eye = core_iris.get_eyelid(foto, self.model)
        return _eye

    def masked_iris_circle(self, img, _radius,_x,_y):
        _height, _width, _ = img.shape
        _black_fon = np.zeros((_height, _width, 3), np.uint8)
        cv2.circle(_black_fon, (_x, _y), _radius, (255, 255, 255), -1)
        _filter = np.array([255, 255, 255], dtype="uint8")
        _mask = cv2.inRange(_black_fon, _filter, _filter)
        _masked = cv2.bitwise_and(img, img, mask=_mask)
        return _masked

    def masked_iris_polygon(self, img, points):
        _height, _width, _ = img.shape
        _black_fon = np.zeros((_height, _width, 3), np.uint8)
        cv2.fillPoly(_black_fon, [np.array(points, np.int32)], (255, 255, 255))
        _filter = np.array([255, 255, 255], dtype="uint8")
        _mask = cv2.inRange(_black_fon, _filter, _filter)
        _masked = cv2.bitwise_and(img, img, mask=_mask)
        return _masked

    def segmented_iris_circle(self, img):
        self.iris = self.get_iris(img)
        _radius = self.iris.radius - 0
        _x = self.iris.center[0] + 3
        _y = self.iris.center[1] + 10
        _seg = self.masked_iris_circle(img, _radius,_x,_y)
        return _seg

    def segmented_iris_polygon(self, img):
        _iris = self.get_iris(img)
        points_iris = _iris.limb_points
        points_iris = self.interpolation(points_iris + [points_iris[0]], method='cubic')
        _seg = self.masked_iris_polygon(img, points_iris)
        return _seg

    def cropped_iris(self, img):
        mask = img != 0
        mask = mask.any(2)
        mask0, mask1 = mask.any(0), mask.any(1)
        return img[np.ix_(mask1, mask0)]

    # Интерполяция
    def interpolation(self, points, method='cubic'):
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        # interpolations_methods = ['slinear', 'quadratic', 'cubic']
        alpha = np.linspace(0, 1, 75)
        interpolator = interp1d(distance, points, kind=method, axis=0)
        return interpolator(alpha)

