# coding: utf-8
import cv2
import numpy as np


class BoundingBox(object):
    """
    cv2 で tracker や Cv2Yolo の返り値などに含まれる tuple の bounding box を
    使いやすく構造体として扱うためのクラス.
    """

    @classmethod
    def distance(cls, bbox_1, bbox_2):
        """
        bounding box 同士のユークリッド距離を返す

        :param BoundingBox bbox_1:
        :param BoundingBox bbox_2:
        :return: bbox 同士のユークリッド距離
        :rtype: float
        """
        if bbox_1.box is None or bbox_2.box is None:
            return np.Infinity
        norm = (bbox_1.center_y - bbox_2.center_y) ** 2. + (bbox_1.center_x - bbox_2.center_x) ** 2.
        return norm ** .5

    def __init__(self, cv2_box):
        """
        :param tuple[int | float] cv2_box: bounding box の tuple. shape = (4, )
            cv2.tracker の update などで帰ってくる tuple はそのまま使用可能です.
            それぞれ (left, top, width, height) である必要があります
        """
        self.box = cv2_box
        if cv2_box is None:
            cv2_box = (None, None, None, None)
        self.left, self.top, self.width, self.height = cv2_box

    @property
    def center_x(self):
        return self.left + self.width / 2

    @property
    def center_y(self):
        return self.top + self.height / 2

    def dist_between(self, x):
        """
        :param BoundingBox x:
        :return:
        """
        return BoundingBox.distance(self, x)


def draw_bbox(img, bbox, color=None, line_width=3):
    """
    画像に対して bounding box の長方形を描く関数

    Args:
        img (np.ndarray):
        bbox (BoundingBox):
        color (iterable[int]):
        line_width (int):

    Returns:
        np.ndarray: 描画後の画像

    """
    drawn = np.copy(img)
    if color is None:
        color = (1, 222, 1)
    cv2.rectangle(drawn,
                  (int(bbox.left), int(bbox.top)),
                  (int(bbox.left + bbox.width), int(bbox.top + bbox.height)),
                  color, line_width)
    return drawn


def draw_text(img, position, text, size=1, color=None):
    """
    :param np.ndarray img: cv2 image.
    :param iterable[int] position:
        記述するテキストの左下の座標. tuple もしくは list で指定.
        shape = (2, )
    :param str text: 描画するテキスト
    :param int size: 文字の大きさ
    :param tuple[int] color: テキストの色. 3次元の int tuple を指定します.
    :return: 描画後の画像
    :rtype: np.ndarray
    """
    drawn = np.copy(img)
    if color is None:
        color = (1, 222, 1)
    position = tuple([int(p) for p in position])
    cv2.putText(drawn, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)
    return drawn
