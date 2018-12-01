# coding: utf-8
"""
simple detection
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import cv2

from nyktk.config import PROCESSED_DIR
from nyktk.cv2_utils import BoundingBox, draw_bbox
from nyktk.detect import CV2YOLODetector


def get_arguments():
    parser = ArgumentParser(__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', help='input image file.', required=True)
    return parser.parse_args()


def main():
    args = get_arguments()
    detector = CV2YOLODetector()
    img_path = args.input
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(args.input)
    results = detector.predict(img, only_person=True)

    out_img = img.copy()
    for idx, prob, box in results:
        bbox = BoundingBox(box)
        out_img = draw_bbox(out_img, bbox)

    out_path = os.path.join(PROCESSED_DIR, '{img_name}_detected.jpg'.format(**locals()))
    print('save to {}'.format(out_path))
    cv2.imwrite(out_path, out_img)


if __name__ == '__main__':
    main()
