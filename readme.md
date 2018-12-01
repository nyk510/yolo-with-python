# Yolo with python

python から YOLO を実行するためのサンプルモジュールの実装です。

opencv-python の DNN から YOLO の重みと設定を読み込み、実行しています。

## Requirements

docker 及び docker-compose による環境作成を推奨しています。

## Quick Start

docker-compose をつかってコンテナを作成し立ち上げます

```bash
cp project.env .env
docker-compose up -d
```

[localhost:4004](http://localhost:4004/tree/notebooks) へアクセスして jupyter へログインします. (password は dolphin です)

## Sample: 画像解析

はじめに docker 内部に潜り込む

```bash
docker exec -it yolo-with-python-app bash
```

```bash
root@fde3d4fa9be4:/home# python simple_detect.py -i sample/maeda.jpg
save to /home/data/processed/maeda_detected.jpg
```

解析結果
![maeda_解析結果](./sample/maeda_detect.png)

### 動画解析

```bash
python movie_person_detection.py -f /home/sample/sample_movie.mp4

# 中略
[INFO video-stream] 2018-10-24 09:50:24,323: save to /home/sample/output_frames/frame_01202.png
[INFO video-stream] 2018-10-24 09:50:24,354: finished
[INFO video-stream] 2018-10-24 09:50:24,355: 0.917 [sec/frame]
```
