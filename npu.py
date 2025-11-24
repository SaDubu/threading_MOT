#yolo5
import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

OBJ_THRESH = 0.35
NMS_THRESH = 0.45
IMG_SIZE = 640

def get_model(model_name, use_core=0x0) :
    rknn = RKNN(verbose=True)
    ret = rknn.load_rknn(model_name)
    if ret != 0 :
        print(f'not_load_{model_name}')
        self.logger.error(f'not_load_{model_name}')
        self.stop_q.put(True)
        return 1

    ret = rknn.init_runtime(target='rk3588', core_mask=use_core)
    if ret != 0 :
        print(f'{model_name} is not init')
        self.logger.error(f'{model_name} is not init')
        self.stop_q.put(True)
        return 1

    print(f"{model_name} init complate")

    return rknn    

class Yolov5() :
    def __init__(self) :
        self.stop_q = None
        self.model = None
        self.rknn = None
        self.frame_q = None
        self.box_q = None
        self.logger = None
        self.no_detect_flag = None

    def set_no_detect_flag(self, q) :
        self.no_detect_flag = q
    
    def set_logger(self, logger_object) :
        self.logger = logger_object
        self.logger.info('yolov5 class object activate')

    def is_stop(self) :
        stop_q_size = self.stop_q.qsize()
        if stop_q_size == 0 :
            return False
        return True

    def set_frame_q(self, q) :
        self.frame_q = q
    
    def set_box_q(self, q) :
        self.box_q = q

    def set_stop_q(self, q) :
        self.stop_q = q

    def set_model(self, name, use_core=0x0) :
        self.model = name
        '''
        # NPU Core Mask
        NPU_CORE_AUTO = 0x0
        NPU_CORE_0 = 0x1
        NPU_CORE_1 = 0x2
        NPU_CORE_2 = 0x4
        NPU_CORE_0_1 = 0x3
        NPU_CORE_0_1_2 = 0x7
        NPU_CORE_ALL = 0xffff

        '''
        self.model_init(use_core)

    def xywh2xyxy(self, x):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


    def process(self, input, mask, anchors):

        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = input[..., 4]
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = input[..., 5:]

        box_xy = input[..., :2]*2 - 0.5

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy *= int(IMG_SIZE/grid_h)

        box_wh = pow(input[..., 2:4]*2, 2)
        box_wh = box_wh * anchors

        box = np.concatenate((box_xy, box_wh), axis=-1)

        return box, box_confidence, box_class_probs


    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        boxes = boxes.reshape(-1, 4)
        box_confidences = box_confidences.reshape(-1)
        box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

        _box_pos = np.where(box_confidences >= OBJ_THRESH)
        boxes = boxes[_box_pos]
        box_confidences = box_confidences[_box_pos]
        box_class_probs = box_class_probs[_box_pos]

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= OBJ_THRESH)

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        scores = (class_max_score* box_confidences)[_class_pos]

        return boxes, classes, scores


    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


    def yolov5_post_process(self, input_data):
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                [59, 119], [116, 90], [156, 198], [373, 326]]

        boxes, classes, scores = [], [], []
        for input, mask in zip(input_data, masks):
            b, c, s = self.process(input, mask, anchors)
            b, c, s = self.filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        boxes = np.concatenate(boxes)
        boxes = self.xywh2xyxy(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self.nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def first_process(self, outputs) :
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
        return input_data

    def model_init(self, use_core) :
        self.rknn = RKNN(verbose=True)

        ret = self.rknn.load_rknn(self.model)
        if ret != 0 :
            print('not_load_rknn_model')
            #self.logger.error('not_load_rknn_model')
            self.stop_q.put(True)
            return 1

        ret = self.rknn.init_runtime(target='rk3588', core_mask=use_core)
        if ret != 0 :
            print('model is not init')
            #self.logger.error('model is not init')
            self.stop_q.put(True)
            return 1

        print("model init complate")
        return 0
    
    def process_for_run(self, img, tracker_input) :
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img2 = np.expand_dims(img2, 0)
        outputs = self.rknn.inference(inputs=[img2], data_format=['nhwc'])
        input_data = self.first_process(outputs)

        boxes, classes, scores = self.yolov5_post_process(input_data)

        if boxes is None :
            return 1

        for box, cla, sco in zip(boxes, classes, scores) :
            x1, y1, x2, y2 = box
            sort_input = [x1, y1, x2, y2, cla, sco]
            tracker_input.append(sort_input)
        
        return 0

    def run(self) :
        img = None
        end = None

        tracker_input = []
        while True :
            if self.is_stop() :
                break

            if self.frame_q.qsize() == 0 :
                continue

            img = self.frame_q.get()
            
            end = self.process_for_run(img, tracker_input)

            if end == 1 :
                self.logger.error('yolo error')
                self.no_detect_flag.put(True)
                continue
            
            self.box_q.put(tracker_input)

            tracker_input = []
                
        return 0 

    def __del__(self) :
        print('\nyolov5 class destructor')
        if self.rknn :
            self.rknn.release()

            