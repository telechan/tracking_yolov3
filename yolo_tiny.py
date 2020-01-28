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
        "model_path": 'model_data/yolo-tiny.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
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

def get_color(image, objects):
    color_list = {}
    for (object_ID, centroid) in objects.items():
        img = image[int(centroid[1]) : int(centroid[1]) + 30, int(centroid[0]) : int(centroid[0]) + 30]
        r = np.floor(img.T[2].flatten().mean()).astype('int32')
        g = np.floor(img.T[1].flatten().mean()).astype('int32')
        b = np.floor(img.T[0].flatten().mean()).astype('int32')
        # print((r, g, b))
        hsv = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]

        if hsv[1] > 50:
            if hsv[2] > 60:
                if hsv[0] < 15 or hsv[0] >= 160:
                    color_list[object_ID] = ('red', hsv)
                elif hsv[0] < 40:
                    color_list[object_ID] = ('orange, yellow', hsv)
                elif hsv[0] < 80:
                    color_list[object_ID] = ('green', hsv)
                elif hsv[0] < 130:
                    color_list[object_ID] = ('blue', hsv)
                elif hsv[0] < 160:
                    color_list[object_ID] = ('purple, pink', hsv)
                else:
                    color_list[object_ID] = ('??', hsv)
            else:
                color_list[object_ID] = ('??', hsv)
        else:
            if hsv[2] > 180:
                color_list[object_ID] = ('white', hsv)
            elif hsv[2] > 120:
                color_list[object_ID] = ('gray', hsv)
            elif hsv[2] <= 120:
                color_list[object_ID] = ('black', hsv)
            else:
                color_list[object_ID] = ('??', hsv)
    return color_list        

def track_objects(image, objects, count1, count2, trackableObjects, color_list):
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

        text = "ID {} {}".format(objectID, color_list[objectID][0])
        text2 = " {}".format(color_list[objectID][1])
        print(text + text2)
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [centroid[0] - 5, centroid[1] -5, centroid[0] + 5, centroid[1] + 5],
            fill=(127, 255, 0)
        )
        draw.text((centroid[0] -10, centroid[1] -25), text, fill=(127, 255, 0), font=font)
        del draw
    return image, count1, count2

def max_min_area(mask, boxes, scores, max_area, min_area):
    for (i, (top, left, bottom, right)) in enumerate(boxes):
        if scores[i] >= 0.20:
            top = top // 3
            left = left // 3
            bottom = bottom // 3
            right = right // 3
            mask1 = mask[top : bottom, left : right]

            max_lim = mask1.shape[1] * mask1.shape[0]
            min_lim = (mask.shape[1] * mask.shape[0]) * 0.03

            contours, hierarchy = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for _, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if max_area < area and max_lim > area:
                    max_area = area
                if min_area > area and min_lim < area:
                    min_area = area
    return max_area, min_area

def get_area(mask, boxes, scores):
    del_list = []
    flag = False
    for (i, box) in enumerate(boxes):
        if scores[i] >= 0.20:
            top = box[0] // 3
            left = box[1] // 3
            bottom = box[2] // 3
            right = box[3] // 3
            mask1 = mask[top : bottom, left : right]

            max_lim = mask1.shape[1] * mask1.shape[0]
            min_lim = (mask.shape[1] * mask.shape[0]) * 0.03

            contours, hierarchy = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                del_list.append(box)
            else:
                for _, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if area >= max_lim or area < min_lim:
                        flag = True
                    else:
                        flag = False
                        break
                if flag:
                    del_list.append(box)
        else:
            del_list.append(box)
    for n in del_list:
        boxes = [box for box in boxes if box != n]
    return boxes

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

    # fgbg = cv2.createBackgroundSubtractorKNN()
    fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
    ct = CentroidTracker(maxDisappeared=20, maxDistance=90)
    trackableObjects = {}
    del_ID = 0
    count_a = 0
    count_b = 0
    flag = False
    max_area = 0
    min_area = video_size[0] * video_size[1]
    area_time = 0
    accum_time = 0
    curr_fps = 0
    max_fps = 0
    fps = "FPS: ??"

    prev_time = timer()
    while True:
        _, frame = vid.read()
        if type(frame) == type(None): break

        resize_img = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        mask = fgbg.apply(resize_img)
        # mask1 = mask
        # thresh = cv2.threshold(mask, 3, 255, cv2.THRESH_BINARY)[1]
        # cv2.namedWindow('maskwindow', cv2.WINDOW_NORMAL)
        # cv2.imshow('maskwindow', mask)

        if flag:
            out_image = frame
            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for _, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > min_area and area < (max_area / 2):
                    flag = False
                    break
        else:
            image = Image.fromarray(frame)
            image, out_boxes, out_scores = yolo.detect_image(image)
            out_boxes = get_area(mask, out_boxes, out_scores)
            # if len(out_boxes) != 0:
            #     for i, box in enumerate(out_boxes):
            #         cv2.rectangle(frame, (box[1] // 3, box[0] // 3), (box[3] // 3, box[2] // 3), (127, 255, 0), thickness=1)
            objects = ct.update(out_boxes)
            color_list = get_color(frame, objects)
            image, count_b, count_a = track_objects(image, objects, count_b, count_a, trackableObjects, color_list)
            out_image = np.asarray(image)
            if len(out_boxes) != 0:
                for i, box in enumerate(out_boxes):
                    cv2.rectangle(out_image, (box[1], box[0]), (box[3], box[2]), (127, 255, 0), thickness=2)

            if len(objects) != 0:
                objects_ID = list(objects.keys())
                trackable_ID = list(trackableObjects.keys())
                non_IDs = list(set(trackable_ID) - set(objects_ID))
                for non_ID in non_IDs:
                    del trackableObjects[non_ID]
                if area_time < 150:
                    max_area, min_area = max_min_area(mask, out_boxes, out_scores, max_area, min_area)
                    area_time += 1
            elif len(objects) == 0 and area_time >= 150:
                flag = True

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
            ("count B", count_b),
            ("coutn A", count_a),
            ("min area", min_area),
            ("max area", max_area)
        ]
        for (i, (k, v)) in enumerate(info):
            textInfo = "{}: {}".format(k, v)
            cv2.putText(out_image, text=textInfo, org=(10, out_image.shape[0] - ((30 * i) + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(127, 255, 0), thickness=1)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out_image)
        # cv2.namedWindow('maskwindow', cv2.WINDOW_NORMAL)
        # cv2.imshow('maskwindow', mask1)

        if isOutput:
            out.write(out_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Max area: {}, Min area: {}".format(max_area, min_area))
    print("Max FPS: {}FPS".format(max_fps))
    yolo.close_session()
