#ov13850 camera mipi port video11 get image 

import cv2
import time

WIDTH = 640
HEIGHT = 480

class Cam :
    def __init__(self) :
        self.stop_q = None
        self.frame = None
        self.cap = None
        self.frame_q = None
        self.fps_q = None
        self.width = WIDTH
        self.height = HEIGHT
        self.logger = None
    
    def set_logger(self, logger_object) :
        self.logger = logger_object
        self.logger.info('cam class object activate')

    def set_frame_size(self, w=None, h=None) :
        if w is not None :
            self.width = w
        if h is not None :
            self.height = h

    def set_cap(self, device) :
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened() :
            return 1
        #fps is limit 15

        return 0
    
    def set_fps_q(self, q) :
        self.fps_q = q

    def set_frame_q(self, q) :
        self.frame_q = q

    def set_stop_q(self, q) :
        self.stop_q = q

    def get_image(self) :
        ret, self.frame = self.cap.read()
        if not ret :
            print()
            print("what?")
            return 1

        return 0

    def is_stop(self) :
        stop_q_size = self.stop_q.qsize()
        if stop_q_size == 0 :
            return False
        return True

    def run(self) :
        count = 0
        frame_get = 0
        fps_q_size = 0
        while True :
            if self.is_stop() :
                break
            ret = self.get_image()
            if ret == 1:
                print("no getting image")
                self.logger.error('no getting image')
                self.stop_q.put(True)
                break
            self.frame = cv2.resize(self.frame, (self.width, self.height))
            #print(self.frame.shape)
            self.frame_q.put(self.frame)
            #print(self.frame_q.qsize())
            count += 1
            fps_q_size = self.fps_q.qsize()
            if fps_q_size != 0:
                self.fps_q.get()
            self.fps_q.put(count)

    def test(self) :
        ret = self.get_image()
        if ret == 1:
            print("SAD__:(")
            return
        cv2.imshow("ff", self.frame)
        cv2.waitKey(1) 

    def __del__(self) :
        print('\ncam class destructor')
        if self.cap :
            self.cap.release()
    