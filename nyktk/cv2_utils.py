# coding: utf-8
from time import time

import cv2
import numpy as np

from .utils import get_logger


class AbstractVideoStream(object):
    """
    動画ファイルを読み込んでフレームごとに何らかの操作を行うロジックの抽象クラス.
    使用する際には update, before_read, after_read などを override して使用してください.
    """

    def __init__(self, video_path, log_level='INFO'):
        """
        Args:
            video_path(str): 読み込む動画への path を指定します.
            log_level(str): logging のレベルを指定します.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.logger = get_logger('video-stream', log_level=log_level)

        self.start_time = None
        self.end_time = None

    def run(self):
        """
        動画読み込みを開始するメソッド

        Returns:
            int: 動画の total count
        """
        frame_counter = 0
        self.before_read()
        self.start_time = time()
        self.logger.info('start loading')
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            frame_counter += 1
            try:
                self.update(frame, frame_counter)
            except StopIteration:
                break
        self.end_time = time()
        self.after_read(frame_counter)
        return frame_counter

    def update(self, frame, frame_counter):
        """
        動画の各フレームに対して呼び出されるメソッドです

        Args:
            frame(np.ndarray):
            frame_counter(int):
        """
        pass

    def before_read(self):
        pass

    def after_read(self, total_frame):
        duration = self.end_time - self.start_time
        time_per_frame = duration / total_frame

        self.logger.info('finished')
        self.logger.info('{:.3f} [sec/frame]'.format(time_per_frame))


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
