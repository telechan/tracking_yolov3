# Car Counting YOLOv3

## Introduction
This is an implementation of the car tracking and counting function to [keras-yolo3](https://github.com/qqwweee/keras-yolo3).
 It used the centroid tracking algorithm of [this article](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/).

## Requirements

My test environment is:
- Python 3.6.7
- tensorflow-gpu 1.12.0
- Keras 2.1.6
- opencv-python 4.1.2.30
- pillow 6.2.1
- matplotlib 3.1.2
- numpy 1.17.4

You can install by running the following code:
```bash
pip install -r requirement.txt
```

## Quick Start

1. Download YOLOv3 (or tiny YOLO) weights from [YOLO website](http://pjreddie.com/darknet/yolo/). Or run the following code:

```bash
wget https://pjreddie.com/media/files/yolov3.weights

# For tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

2. Convert the Darknet YOLO weights to Keras weights.

```bash
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

# For tiny
python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo-tiny.h5
```

3. Run YOLO detection.

Run with images:
```bash
python yolo_video.py --image

# For tiny
python yolo_video_tiny.py --image
```

Then enter your image path. The result image will be output to the same directory.
```bash
Image detection mode
 Ignoring remaining command line arguments: ./path2your_video,
2019-11-20 00:28:38.457780: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
model_data/yolo.h5 model, anchors, and classes loaded.
Input image filepath:
```
---

Run with videos:
```bash
python yolo_video.py --input [your video path] --output [output path (optional)]

# For tiny
python yolo_video_tiny.py --input [your video path] --output [output path (optional)]
```

If you want to stop it type 'q'.

## Usage

Use --help to see usage of yolo_video.py:
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

MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).