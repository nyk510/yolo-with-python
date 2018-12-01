"""
Movie Person Detection

動画から人領域を切り出してフレームごとに保存していくスクリプト
"""

import argparse
import os
from datetime import datetime

import cv2
import imutils
import numpy as np

from nyktk.config import PROCESSED_DIR
from nyktk.cv2_utils import AbstractVideoStream, draw_bbox, BoundingBox
from nyktk.detect import CV2YOLODetector


class SimplePersonDetectVideoStream(AbstractVideoStream):
    def __init__(self, video_path, output_dir, max_frames=300):
        self.detector = CV2YOLODetector()
        self.output_dir = os.path.join(PROCESSED_DIR, output_dir)
        self.max_frames = max_frames
        super(SimplePersonDetectVideoStream, self).__init__(video_path)

    def before_read(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def update(self, img, frame_counter):
        if frame_counter > self.max_frames:
            raise StopIteration()

        img = imutils.resize(img, width=400)
        detections = self.detector.predict(img)

        self.logger.debug('detect num:{}'.format(len(detections)))
        draw_img = np.copy(img)
        for _, prob, area in detections:
            bbox = BoundingBox(area)
            draw_img = draw_bbox(draw_img, bbox)
        out_path = os.path.join(self.output_dir, 'frame_{:05d}.png'.format(frame_counter))
        self.logger.info('save to {}'.format(out_path))
        cv2.imwrite(out_path, draw_img)


def parse_arguments():
    parser = argparse.ArgumentParser(__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', required=True, type=str)
    parser.add_argument('-o', '--out', help='output dir name', type=str, default='movie_detect')
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_dir = args.out
    now = datetime.now().strftime('%Y-%m-%d_%H:%m')
    output_dir = '{output_dir}_{now}'.format(**locals())
    stream = SimplePersonDetectVideoStream(video_path=args.file, output_dir=output_dir)
    stream.run()


if __name__ == '__main__':
    main()
