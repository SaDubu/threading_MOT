import threading
import queue
import time
import cv2
import sys
import select

#add me
from cam import Cam
from npu import Yolov5
from sort import Sort
from draw import Draw
from thread import WatchThread
from log_mointer import Monitor

DEVICE_PORT = "/dev/video11"
MODEL_NAME = "yolo.rknn"

def run() :
    #camera, canvas, main - put, all - get
    stop_queue = queue.Queue()
    #camera - put, yolo - get
    frame_queue = queue.Queue()
    #yolo - put, track - get
    box_queue = queue.Queue()
    #track - put, canvas - get
    track_queue = queue.Queue()
    #yolo - put, canvas - get 
    copy_frame_queue = queue.Queue()
    #camera - get, put, monitor - get
    fps_queue = queue.Queue()
    #canvas - put, main - get
    drawed_queue = queue.Queue()

    #object setting
    camera = Cam()
    camera.set_stop_q(stop_queue)
    camera.set_cap(DEVICE_PORT)
    camera.set_frame_q(frame_queue)
    camera.set_frame_size(640, 640)
    camera.set_fps_q(fps_queue)

    yolo = Yolov5()
    yolo.set_model(MODEL_NAME)
    yolo.set_frame_q(frame_queue)
    yolo.set_box_q(box_queue)
    yolo.set_copy_frame_q(copy_frame_queue)
    yolo.set_stop_q(stop_queue)

    canvas = Draw()
    canvas.set_frame_q(copy_frame_queue)
    canvas.set_track_q(track_queue)
    canvas.set_stop_q(stop_queue)
    canvas.set_draw_q(drawed_queue)

    track = Sort(
        max_age=3,
        min_hits=1,
        iou_threshold=0.3
    )
    track.set_box_q(box_queue)
    track.set_track_q(track_queue)
    track.set_stop_q(stop_queue)

    monitor = Monitor()
    monitor.set_stop_q(stop_queue)
    monitor.set_fps_q(fps_queue)

    #thread setting
    threads = []

    cam_thread = WatchThread(target=camera.run, name='cam')
    yolo_thread = WatchThread(target=yolo.run, name='yolo')
    track_thread = WatchThread(target=track.run, name='track')
    canvas_thread = WatchThread(target=canvas.run, name='canvas')
    monitor_thread = WatchThread(target=monitor.run, name='monitor')

    threads.append(cam_thread)
    threads.append(yolo_thread)
    threads.append(track_thread)
    threads.append(canvas_thread)
    threads.append(monitor_thread)

    frame_queue_size = 0
    box_queue_size = 0
    track_queue_size = 0
    copy_frame_queue_size = 0
    drawed_queue_size = 0
    
    q_size_sum = 0
    threads_lenth = len(threads)

    emergency_stop = False

    output_img = None
    
    #main process
    while True:
        if stop_queue.qsize() != 0:
            break
        
        for i in range(threads_lenth):
            threads[i].restart_if_out()

        frame_queue_size = frame_queue.qsize()
        box_queue_size = box_queue.qsize()
        track_queue_size = track_queue.qsize()
        copy_frame_queue_size = copy_frame_queue.qsize()
        drawed_queue_size = drawed_queue.qsize()

        q_size_sum = frame_queue_size + box_queue_size + track_queue_size + copy_frame_queue_size + drawed_queue_size

        if q_size_sum > 300 :
            stop_queue.put(True)
            emergency_stop = True
            break

        monitor.update("queue_camera_in", frame_queue_size)
        monitor.update("queue_detect", box_queue_size)
        monitor.update("queue_track", track_queue_size)
        monitor.update("queue_capture", copy_frame_queue_size)

        try :
            output_img = drawed_queue.get(timeout=1)
        except queue.Empty:
            continue
        cv2.imshow("show", output_img)

        input_key = cv2.waitKey(1) & 0xFF

        if input_key == ord('q') :
            print("\n\n\033[1;31m⚠️  [EXIT] Termination command 'q' received. Stopping threads...\033[0m\n")
            stop_queue.put(True)
            break
            
    cv2.destroyAllWindows()
    #thread close
    for i in range(threads_lenth):
        threads[i].thread_end()

    return emergency_stop

def main() :
    is_run_fail = False
    while True :
        time.sleep(10)
        is_run_fail = run()
        if not is_run_fail :
            break
        
        print()
        print('The program encountered a problem while running.')
        print('Please check the camera status.')
        print('Restarting...')
        print()
        print('-'*50)
        print()

    print('close')
    return 0

if __name__ == "__main__" :
    main()