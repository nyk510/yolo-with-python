# coding: utf-8
"""
utility functions
"""
import math
import os
from logging import getLogger, StreamHandler, FileHandler, Formatter
from PIL import Image
import requests
from scipy import spatial
from tqdm import tqdm


def get_logger(name, log_level="DEBUG", output_file=None, handler_level="INFO"):
    """
    logger を取得する function
    more infomation, see https://docs.python.jp/3/howto/logging.html

    :param str name: logger の名前
    :param str log_level: 
        logger 自体の出力の度合い.
        指定されたレベル以上のログのみを handler に出力する.
        default は `"DEBUG"`
    :param str | None output_file:
        ログを出力するファイルの名前. None の時はファイル出力を行わない.
    :param str handler_level:
        logger handler がログを送出する level.
        これ以上のレベルのログのみ handler が処理を行う.
    :return: logger
    """
    logger = getLogger(name)

    formatter = Formatter("[%(levelname)s %(name)s] %(asctime)s: %(message)s")

    handler = StreamHandler()
    logger.setLevel(log_level)
    handler.setLevel(handler_level)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if output_file:
        file_handler = FileHandler(output_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(handler_level)
        logger.addHandler(file_handler)

    return logger

logger = get_logger(__name__)


def download_file(url, save_to):
    """
    url で与えられたファイルを特定のパスに保存する method
    :param str url: 保存するファイルの url
    :param str save_to: 保存先への path
    :return:
    """
    if os.path.exists(save_to):
        logger.info('weight already exists')
        return

    dir_path = os.path.dirname(save_to)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open('output.bin', 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    return
