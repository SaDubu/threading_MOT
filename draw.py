import cv2
import numpy as np
import time
import queue

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

class Draw() :
    def __init__(self) :
        self.track_q = None
        self.frame_q = None
        self.stop_q = None
        self.drawed_q = None
        self.logger = None
        self.no_detect_flag = None

    def set_no_detect_flag(self, q) :
        self.no_detect_flag = q
    
    def set_logger(self, logger_object) :
        self.logger = logger_object
        self.logger.info('draw class object activate')
    
    def set_draw_q(self, q) :
        self.drawed_q = q

    def set_frame_q(self, q) :
        self.frame_q = q

    def set_track_q(self, q) :
        self.track_q = q

    def set_stop_q(self, q) :
        self.stop_q = q

    def is_stop(self) :
        stop_q_size = self.stop_q.qsize()
        if stop_q_size == 0 :
            return False
        return True

    def draw(self, image, boxes, scores, classes, track_id):
        """
        Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        #print("{:^12} {:^12}  {}".format('class', 'score', 'xmin, ymin, xmax, ymax'))
        #print('-' * 50)
        for box, score, cl, trk in zip(boxes, scores, classes, track_id):
            top, left, right, bottom = box
            top = int(top)
            left = int(left)
            right = int(right)
            bottom = int(bottom)
            cl = int(cl)
            id_n = str(int(trk // 1))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f} ID : {2:^4}'.format(CLASSES[cl], score, id_n),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            #print("{:^9} {:^2} {:^12.3f} [{:>4}, {:>4}, {:>4}, {:>4}]".format(CLASSES[cl], id_n, score, top, left, right, bottom))
    
    def post_process(self, det, box, class_id, track_id, score) :
        det_len = len(det)
        if det_len == 0 :
            return None, None, None, None

        for track in det :
            box.append(track[0])
            class_id.append(track[2])
            track_id.append(track[1])
            score.append(track[3])
        
        return box, class_id, track_id, score
        
    def no_detect_img(self) :
        try :
            img = self.frame_q.get_nowait()
        except queue.Empty:
            time.sleep(0.002)
            return 0
        
        text = "NO DETECT"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        color = (0, 0, 255)

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        img_h, img_w = img.shape[:2]
        x = (img_w - text_w) // 2
        y = (img_h - text_h) // 2

        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

        self.drawed_q.put(img)
        return 0

    def run(self) :
        img = None
        track = None

        box = []
        class_id = []
        track_id = []
        score = []

        while True :
            if self.is_stop() :
                break
            
            if self.no_detect_flag.qsize() != 0 :
                self.no_detect_flag.get()
                self.no_detect_img()
                continue

            try :
                track = self.track_q.get_nowait()
            except queue.Empty:
                time.sleep(0.002)
                continue

            try :
                img = self.frame_q.get_nowait()
            except queue.Empty:
                time.sleep(0.002)
                continue

            box = []
            class_id = []
            track_id = []
            score = []
    
            box, class_id, track_id, score = self.post_process(track, box, class_id, track_id, score)

            if box is None :
                continue

            self.draw(img, box, score, class_id, track_id)

            self.drawed_q.put(img)

            img = None
            track = None
        return 0

    def __del__(self):
        print('\ndraw class destructor')