"""
SORT: A Simple, Online and Realtime Tracker
Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

import cv2
import queue

np.random.seed(0)

# add me
MAX_ID = 100

RET_ID_INDEX_NUM = 4

CLS_ID_INDEX_NUM = 5

is_up = False

def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0
    global is_up

    def __init__(self, bbox, class_id=0, score=0):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        if is_up :
            KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.cls_id = int(class_id)
        self.score = float(score)
        self.cos_sim = None

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 6), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    output = dot / (norm_a * norm_b)
    #print(output)
    return output

def use_npu_cos_sim(a, b, model):
    output = model.inference(inputs=[a, b], data_format=None)
    #print(output)
    return output[0][0]

def use_cnn(img, model):
    """
    I plan to use the cnn model for feature extraction and then test whether the objects are identical
    """
    cnn_input_size = (224, 224)
    img1 = cv2.resize(img, cnn_input_size)
    outputs = model.inference(inputs=[img1], data_format=["nhwc"])
    outputs = np.array(outputs)
    return outputs


def crop_roi(image, bbox):
    """
    image: numpy array (H, W, C)
    bbox: list or tuple [x1, y1, x2, y2]
    return: Cropped image region (numpy array)
    """
    x1, y1, x2, y2 = bbox

    # Clamp coordinates to stay within the image boundaries.
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image.shape[1], int(x2))
    y2 = min(image.shape[0], int(y2))

    return image[y1:y2, x1:x2]

def is_empty(img):
    # None + numpy array size
    if img is None:
        return True
    if isinstance(img, np.ndarray) and img.size == 0:
        return True
    return False

class Sort_addme(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

        # To be used for loading the model.
        self.model = None
        self.cos_model = None

        self.now_frame = None
        self.past_frame = None
        self.t_1_trk = None
        
        self.ret_len = 0
        self.t_1_len = 0

        self.stop_q = None
        self.box_q = None
        self.track_q = None
        self.sort_frame_q = None

        self.logger = None

    def set_sort_frame_q(self, q) :
        self.sort_frame_q = q
        
    def set_logger(self, logger_object) :
        self.logger = logger_object
        self.logger.info('sort_addme class object activate')
    
    def set_stop_q(self, q) :
        self.stop_q = q

    def set_box_q(self, q) :
        self.box_q = q

    def set_track_q(self, q) :
        self.track_q = q

    def is_stop(self) :
        stop_q_size = self.stop_q.qsize()
        if stop_q_size == 0 :
            return False
        return True

    def set_model(self, model):
        self.model = model
    
    def set_cos_model(self, model):
        self.cos_model = model

    def use_cnn(self, img1, img2, model):
        use_cnn(img1, img2, model)

    def assign_new_id(self, new_class_id):
        existing_ids = [trk.id for trk in self.trackers if trk.cls_id == new_class_id]
        existing_ids = sorted(existing_ids)
        for i, eid in enumerate(existing_ids, start=1):
            if i != eid:
                return i
        return existing_ids[-1] + 1 if existing_ids else 1

    def update_with_cnn(self, dets=np.empty((0, 6)), frame=None, alpha=0.45, is_cpu=True):
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        self.now_frame = frame

        t_1_d = None
        t_1_vec = None

        if self.past_frame is None:
            self.past_frame = frame

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], dets[i, 4], dets[i, 5])
            trk.id = self.assign_new_id(trk.cls_id)
            self.trackers.append(trk)
        i = len(self.trackers)                
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # print(trk.id, trk.cls_id, trk.get_state())
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits) and (trk.cos_sim is None or trk.cos_sim >= alpha):
                ret.append(
                    np.concatenate((d, [trk.id], [trk.cls_id], [trk.score], [trk.cos_sim])).reshape(
                        1, -1
                    )
                )  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        self.ret_len = len(ret)
        
        if self.t_1_trk is not None :
            #start = time.time()
            #print(self.ret_len)
            #print(self.t_1_len)
            for num in range(0, self.ret_len) :
                t = ret[num][0]
                for i in range(0, self.t_1_len) :
                    t_1 = self.t_1_trk[i][0]
                    #print(f't is {t}, t_1 is {t_1}')
                    t_cls = t[CLS_ID_INDEX_NUM]
                    t_1_cls = t_1[CLS_ID_INDEX_NUM]
                    #print(f't_cls is {t_cls}, t_1_cls {t_1_cls}')
                    if t_cls != t_1_cls :
                        continue
                    t_id = t[RET_ID_INDEX_NUM]
                    t_1_id = t_1[RET_ID_INDEX_NUM]
                    #print(f't_id is {t_id}, t_1_id {t_1_id}')
                    if t_id != t_1_id :
                        continue
                    t_d = t[0:4]
                    t_1_d = t_1[0:4]

                    #start = time.time()
                    t_img = crop_roi(self.past_frame, t_d)
                    t_1_img = crop_roi(self.now_frame, t_1_d)
                    #end = time.time()
                    #print(f'{(end - start) * 1000:.2f}msec')

                    if is_empty(t_img) or is_empty(t_1_img):
                        continue
                    
                    #start = time.time()
                    t_vec = use_cnn(t_img, self.model)
                    t_1_vec = use_cnn(t_1_img, self.model)
                    #end = time.time()
                    #print(f'{(end - start) * 1000:.2f}msec')
                    
                    #start = time.time()
                    if is_cpu :
                        self.trackers[i].cos_sim = cosine_sim(t_1_vec,t_vec)
                        continue
                    self.trackers[i].cos_sim = use_npu_cos_sim(t_1_vec,t_vec, self.cos_model)
                    #end = time.time()
                    #print(f'{(end - start) * 1000:.2f}msec')
            #end = time.time()
            #print(f'{(end - start) * 1000:.2f}msec')
                    
        self.t_1_trk = ret
        self.t_1_len = self.ret_len

        self.past_frame = None 
        self.past_frame = frame
        if self.ret_len > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
        
        # At least 4 frames are required, and results start being drawn on the screen from the 4th frame.

        # The code below is designed so that t_2 gets a value when the 3rd frame arrives.
        if self.t_1_trk is not None:
            self.t_2_trk = self.t_1_trk

        # The code below is designed so that t_1 gets a value when the 2nd frame arrives.
        self.t_1_trk = self.trackers

        self.past_frame = frame
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


    def update(self, dets=np.empty((0, 6))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], dets[i, 4])
            trk.id = self.assign_new_id(trk.cls_id)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # print(trk.id, trk.cls_id, trk.get_state())
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id], [trk.cls_id], [trk.score])).reshape(
                        1, -1
                    )
                )  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def test_update(self, dets=np.empty((0, 5)), frame=None, alpha=0.45):
        global is_up
        is_up = True
        
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        self.now_frame = frame

        t_1_d = None
        t_1_vec = None

        if self.past_frame is None:
            self.past_frame = frame

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits) and (trk.cos_sim is None or trk.cos_sim >= alpha):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
                i -= 1
                # remove dead tracklet
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            
            self.ret_len = len(ret)
        
        if self.t_1_trk is not None and self.past_frame is not None:
            for num in range(0, self.ret_len) :
                t = ret[num][0]
                for i in range(0, self.t_1_len) :
                    t_1 = self.t_1_trk[i][0]
                    t_id = t[RET_ID_INDEX_NUM]
                    t_1_id = t_1[RET_ID_INDEX_NUM]
                    if t_id != t_1_id :
                        continue
                    t_d = t[0:4]
                    t_1_d = t_1[0:4]

                    t_img = crop_roi(self.past_frame, t_d)
                    t_1_img = crop_roi(self.now_frame, t_1_d)

                    if is_empty(t_img) or is_empty(t_1_img):
                        continue
                    
                    t_vec = use_cnn(t_img, self.model)
                    t_1_vec = use_cnn(t_1_img, self.model)
                    
                    self.trackers[i].cos_sim = cosine_sim(t_1_vec,t_vec)
                
                    
        self.t_1_trk = ret
        self.t_1_len = self.ret_len

        self.past_frame = frame
        if self.ret_len > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def run(self) :
        raw_data = []
        np_data = None
        result = None
        frame = None

        track_q_input = []

        while True :
            if self.is_stop() :
                break

            try :
                raw_data = self.box_q.get_nowait()
            except queue.Empty:
                time.sleep(0.005)
                continue

            try :
                frame = self.sort_frame_q.get_nowait()
            except queue.Empty:
                time.sleep(0.005)
                continue
            
            np_data = np.asarray(raw_data, dtype=np.float32)

            result = self.update_with_cnn(np_data, frame)
            
            for data in result :
                x1, y1, x2, y2, t_id, t_cla, t_sco, t_cos_sim = data

                if t_cos_sim is None :
                    t_cos_sim = t_sco
                
                #use tuple
                track_q_input.append(([x1, y1, x2, y2], t_id, t_cla, t_cos_sim))
            self.track_q.put(track_q_input)
            track_q_input = [] # <- clear를 하면 queue에 담긴 것도 지워짐. 
            
        return 0

    def __del__(self) :
        print('\nsort_addme class destructor')  


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument(
        "--display",
        dest="display",
        help="Display online tracker output (slow) [False]",
        action="store_true",
    )
    parser.add_argument(
        "--seq_path", help="Path to detections.", type=str, default="data"
    )
    parser.add_argument(
        "--phase", help="Subdirectory in seq_path.", type=str, default="train"
    )
    parser.add_argument(
        "--max_age",
        help="Maximum number of frames to keep alive a track without associated detections.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min_hits",
        help="Minimum number of associated detections before track is initialised.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--iou_threshold", help="Minimum IOU for match.", type=float, default=0.2
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if display:
        if not os.path.exists("mot_benchmark"):
            print(
                "\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n"
            )
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect="equal")

    if not os.path.exists("output"):
        os.makedirs("output")
    pattern = os.path.join(args.seq_path, phase, "*", "det", "det.txt")
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort_addme(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
        )  # create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=",")
        seq = seq_dets_fn[pattern.find("*") :].split(os.path.sep)[0]

        with open(os.path.join("output", "%s.txt" % (seq)), "w") as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if display:
                    fn = os.path.join(
                        "mot_benchmark", phase, seq, "img1", "%06d.jpg" % (frame)
                    )
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + " Tracked Targets")

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(
                        "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                        % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                        file=out_file,
                    )
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(
                            patches.Rectangle(
                                (d[0], d[1]),
                                d[2] - d[0],
                                d[3] - d[1],
                                fill=False,
                                lw=3,
                                ec=colours[d[4] % 32, :],
                            )
                        )

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print(
        "Total Tracking took: %.3f seconds for %d frames or %.1f FPS"
        % (total_time, total_frames, total_frames / total_time)
    )

    if display:
        print("Note: to get real runtime results run without the option: --display")
