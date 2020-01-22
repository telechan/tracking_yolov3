# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
import cv2

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        out_boxes2 = []
        out_scores2 = []
        out_classes2 = []
        for i, c in enumerate(out_classes):
            if c == 2:
                out_boxes2.append(out_boxes[i])
                out_scores2.append(out_scores[i])
                out_classes2.append(out_classes[i])

        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #         size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        if len(out_classes2) != 0:
            for i, c in reversed(list(enumerate(out_classes2))):
                predicted_class = self.class_names[c]
                box = out_boxes2[i]
                score = out_scores2[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                # label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                out_boxes2[i] = [top, left, bottom, right]
                
                # print(label, (left, top), (right, bottom))

                # My kingdom for a good redistributable image drawing library.
                # for i in range(thickness):
                #     draw.rectangle(
                #         [left + i, top + i, right - i, bottom - i],
                #         outline=(127, 255, 0))

        end = timer()
        # print(end - start)
        print('--------------------')
        return image, out_boxes2, out_scores2

    def close_session(self):
        self.sess.close()

def count_line(width, height ,x):
    y = int(((height - (height / 3.4)) / width) * x) + int(height / 3.4)
    return y     

def track_objects(image, objects, count1, count2, trackableObjects):
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            # x = [c[0] for c in to.centroids]
            # y = [c[1] for c in to.centroids]

            # direction_x = centroid[0] - np.mean(x)
            # direction_y = centroid[1] - np.mean(y)

            # to.centroids.append(centroid)
            if not to.counted:
                first_y = count_line(image.size[0], image.size[1], centroid[0])
                now_y = count_line(image.size[0], image.size[1], to.centroids[0][0])

                if centroid[1] < first_y:
                    if to.centroids[0][1] > now_y:
                        count1 += 1
                        to.counted = True
                elif centroid[1] > first_y:
                    if to.centroids[0][1] < now_y:
                        count2 += 1
                        to.counted = True

        trackableObjects[objectID] = to

        text = "ID {}".format(objectID)
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [centroid[0] - 5, centroid[1] -5, centroid[0] + 5, centroid[1] + 5],
            fill=(127, 255, 0)
        )
        draw.text((centroid[0] -10, centroid[1] -25), text, fill=(127, 255, 0), font=font)
        del draw
    return image, count1, count2

def detect_video(yolo, video_path, output_path=""):
    if video_path.isdigit():
        video_path = int(video_path)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # print("input video size: {}".format(video_size))

    isOutput = True if output_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    ct = CentroidTracker(maxDisappeared=20, maxDistance=90)
    trackableObjects = {}
    del_ID = 0
    to_left = 0
    to_right = 0
    area_time = 0
    accum_time = 0
    curr_fps = 0
    max_fps = 0
    fps = "FPS: ??"

    prev_time = timer()
    while True:
        _, frame = vid.read()
        if type(frame) == type(None): break
        image = Image.fromarray(frame)
        image, out_boxes, out_scores = yolo.detect_image(image)
        objects = ct.update(out_boxes)
        image, to_left, to_right = track_objects(image, objects, to_left, to_right, trackableObjects)
        out_image = np.asarray(image)

        if len(objects) != 0:
            objects_ID = list(objects.keys())
            trackable_ID = list(trackableObjects.keys())
            non_IDs = list(set(trackable_ID) - set(objects_ID))
            for non_ID in non_IDs:
                del trackableObjects[non_ID]

        cv2.line(out_image, (0, count_line(out_image.shape[1], out_image.shape[0], 0)), (out_image.shape[1], count_line(out_image.shape[1], out_image.shape[0], out_image.shape[1])), color=(127, 255, 0), thickness=3)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            if max_fps < curr_fps:
                max_fps = curr_fps
            curr_fps = 0
        print(fps)
        cv2.putText(out_image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(127, 255, 0), thickness=2)

        info = [
            ("to left", to_left),
            ("to right", to_right),
        ]
        for (i, (k, v)) in enumerate(info):
            textInfo = "{}: {}".format(k, v)
            cv2.putText(out_image, text=textInfo, org=(10, out_image.shape[0] - ((30 * i) + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(127, 255, 0), thickness=1)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out_image)

        if isOutput:
            out.write(out_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Max FPS: {}FPS".format(max_fps))
    yolo.close_session()
