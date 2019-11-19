# Tracking YOLOv3

## Introduction
This is an implementation of the tracking function to [keras-yolo3](https://github.com/qqwweee/keras-yolo3).
 It used the centroid tracking algorithm of [this article](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/).

---

## Requirements

My test environment is:
- Python 3.6.7
- tensorflow 1.14.0
- Keras 2.3.1
- opencv-python 4.1.1.26
- pillow 6.2.1
- matplotlib 3.1.1
- numpy 1.17.4

---

## Installation

1. Download YOLOv3 (or tiny YOLO) weights from [YOLO website](http://pjreddie.com/darknet/yolo/). Or run the following code.

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

3. 