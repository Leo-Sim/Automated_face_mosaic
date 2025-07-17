
import numpy as np
import random

class DetectionInfo:
    def __init__(self, bbox):

        x1, y1, x2, y2 = bbox

        if not isinstance(x1, int):
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

        self._id = random.randint(10000, 99999)

        self._bbox = [x1, y1, x2, y2]
        self._center_point = self._get_center_point(self._bbox)

    def _get_center_point(self, bbox):
        x1, y1, x2, y2 = bbox

        x = x2 - x1
        y = y2 - y1

        return x, y

    @property
    def id(self):
        return self._id

    @property
    def bbox(self):
        return self._bbox

    @property
    def center_point(self):
        return self._center_point

    @staticmethod
    def get_distance_of_two_points(a: 'DetectionInfo', b: 'DetectionInfo'):

        p1 = np.array(a.center_point)
        p2 = np.array(b.center_point)

        return np.linalg.norm(p2 - p1)

class MosaicInfo(DetectionInfo):

    def __init__(self, bbox, confidence, need_mosaic):
        super().__init__(bbox)

        self._need_mosaic = need_mosaic
        self.confidence = confidence

    @property
    def need_mosaic(self):
        return self._need_mosaic

    @need_mosaic.setter
    def need_mosaic(self, need_mosaic):
        self._need_mosaic = need_mosaic

class FaceDetectionInfo(DetectionInfo):
    def __init__(self, bbox, is_same):
        super().__init__(bbox)

        self._is_same = is_same

    @property
    def is_same(self):
        return self._is_same


