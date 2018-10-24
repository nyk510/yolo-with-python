import cv2
import numpy as np
import os

from .utils import download_file, get_logger
from .config import YOLO_CONFIGS

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


class CV2YOLODetector(object):
    """
    cv2 の dnn モジュールを用いて yolo を実行する detector

    ## requirements
    * 使いたい yolo の重みと設定ファイル (.cfg)

    ## Usage

    先に用意した重みと config ファイルへのパスを instance 生成時に渡します.
    その後 cv2 形式 (BGR) の画像を `predict` で渡します.

    ```python
    detector = CV2YOLODetector()
    img = cv2.imread('sample.jpg')
    detector.predict(img)
    ```
    """
    weight_dir = '/home/weights'

    def __init__(self, model='YOLOv3-416', force_download=False):
        """
        """
        model_config = YOLO_CONFIGS.get(model, None)
        if model_config is None:
            raise ValueError('invalid model name')

        self.model_config = model_config
        self.weight_path = os.path.join(CV2YOLODetector.weight_dir, model + '.weight')
        self.conf_path = os.path.join(CV2YOLODetector.weight_dir, model + '.cfg')
        self.input_shape = model_config.get('input_shape', None)
        self.logger = get_logger('cv2-yolo')

        self._prepare_setting_files(force_download)

        self.net = cv2.dnn.readNet(self.weight_path, self.conf_path)
        self.output_layers = get_output_layers(self.net)

    def _prepare_setting_files(self, force=False):
        weight_url = self.model_config.get('weight', None)
        config_url = self.model_config.get('config', None)
        for url, local_path in zip((weight_url, config_url,), (self.weight_path, self.conf_path)):
            if not force and os.path.exists(local_path):
                continue
            self.logger.info('download from {}'.format(url))
            download_file(url, local_path)

    def predict(self, img, min_confidence=.5, nms_threshold=.4, only_person=True):
        """
        物体領域検知を実行する
        :param img: 検知対象の画像. cv2 で読み込んだ画像である必要がある (i.e. BGR)
        :param float min_confidence: 予測されたクラス確率の最小値. これを下回る確率の物体は検知されない.
        :param float nms_threshold:
            non-maximum-suppression を実行する際の IOU のしきい値.
            0に近い値になるほど一部でも領域がかぶるとひとつの物体として grouping するようになる.
            反対に1に近づくと領域が重なっていても異なる物体とみなすようになる
        :param bool only_person: `True` のとき人クラスのみを考慮する
        :return: クラスid, クラスの確率, bounding box の tuple の list
            bounding-box は left, top, width, height の順
            物体が検知されたなかった場合には空のリスト `[]` を返します
        """
        blob = cv2.dnn.blobFromImage(img, 1. / 255, self.input_shape, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        img_height, img_width = img.shape[:2]

        # initialization
        class_ids = []
        confidences = []
        boxes = []

        for detections in outputs:
            for detection in detections:
                scores = detection[5:]

                # 人クラスは id = 0 なので only person のとき 0 以外が最大値となる時無視する
                class_id = np.argmax(scores)
                if only_person and class_id != 0:
                    continue

                confidence = scores[class_id]
                if confidence < min_confidence:
                    continue

                center_x = detection[0] * img_width
                center_y = detection[1] * img_height
                w = detection[2] * img_width
                h = detection[3] * img_height
                left = min(center_x - w / 2, img_width - w)
                top = min(center_y - h / 2, img_height - h)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, w, h])

        if len(confidences) == 0:
            return []

        # Non Maximum Supression を行って bbox の数を賢く減らす
        indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)
        results = [(class_ids[idx], confidences[idx], boxes[idx]) for idx in indices.reshape(-1)]
        return results