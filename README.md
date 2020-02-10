# Car Counting YOLOv3

## Introduction
[keras-yolo3](https://github.com/qqwweee/keras-yolo3)をベースに車両の追跡・カウントを行うプログラムです．
追跡手法には[こちらのブログ記事](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/)の内容を参考にしました．

## 実行環境
実行に使ったPC環境は
|||
|:---:|:---:|
|**OS**|Ubuntu|
|**CPU**|Intel Core i7-8700|
|**GPU**|GeForce GTX 1050 Ti|

実行環境に使用したパッケージ一覧:
- Python 3.6.7
- tensorflow-gpu 1.12.0
- Keras 2.1.6
- opencv-python 4.1.2.30
- pillow 6.2.1
- matplotlib 3.1.2
- numpy 1.17.4

以下のコードでパッケージを一括インストールできます．
```bash
pip install -r requirement.txt
```
Keras・tensorflowのバージョンはcudaのバージョンによって変わることに注意．

## Quick Start
1. YOLOの学習済み重みを[YOLOのサイト](http://pjreddie.com/darknet/yolo/)からダウンロードしてくる. もしくは以下のコードでダウンロードする．
```bash
wget https://pjreddie.com/media/files/yolov3.weights

# tiny版
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

2. Darknet YOLOの重みをKeras用にコンバートする.
```bash
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

# tiny版
python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo-tiny.h5
```

3. YOLOの実行

画像での実行:
```bash
python yolo_video.py --image

# tiny版
python yolo_video_tiny.py --image
```
実行すると以下のような表示がされるので，使用したい画像のパスを入力してenterを押せば処理が行われます．
```bash
Image detection mode
Ignoring remaining command line arguments: ./path2your_video,
2019-11-20 00:28:38.457780: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
model_data/yolo.h5 model, anchors, and classes loaded.
Input image filepath:
```
---
動画での実行:
```bash
python yolo_video.py --input [your video path] --output [output path (optional)]

# tiny版
python yolo_video_tiny.py --input [your video path] --output [output path (optional)]
```

"q" を押せば実行を中断できます．

## Usage
yolo_video.py で "--help" オプションを使うと以下のオプション詳細が確認できます．
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
