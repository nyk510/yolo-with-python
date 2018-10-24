import argparse
import os

import cv2
import imutils
import numpy as np

from nyktk.cv2_utils import AbstractVideoStream, draw_bbox, BoundingBox
from nyktk.detect import CV2YOLODetector


class SimplePersonDetectVideoStream(AbstractVideoStream):
    def __init__(self, video_path):
        self.detector = CV2YOLODetector()
        self.output_dir = '/home/sample/output_frames'
        super(SimplePersonDetectVideoStream, self).__init__(video_path)

    def before_read(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def update(self, frame, frame_counter):
        frame = imutils.resize(frame, width=400)
        detections = self.detector.predict(frame)

        self.logger.debug('detect num:{}'.format(len(detections)))
        draw_img = np.copy(frame)
        for _, prob, area in detections:
            bbox = BoundingBox(area)
            draw_img = draw_bbox(draw_img, bbox)
        out_path = os.path.join(self.output_dir, 'frame_{:05d}.png'.format(frame_counter))
        self.logger.info('save to {}'.format(out_path))
        cv2.imwrite(out_path, draw_img)


def parse_arguments():
    parser = argparse.ArgumentParser('movie person detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    stream = SimplePersonDetectVideoStream(video_path=args.file)
    stream.run()


if __name__ == '__main__':
    main()
