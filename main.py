import threading
import queue
import time
import cv2
import sys
import select
import argparse
import tracemalloc

#add me
from cam import Cam
from npu import Yolov5, get_model
from sort import Sort
from draw import Draw
from thread import WatchThread
from log_mointer import Monitor, Logger

from sort_addme import Sort_addme

DEVICE_PORT = "/dev/video11"
MODEL_NAME = "yolo.rknn"
SORT_CNN_MODEL_NAME = 'resnet18_outdim_10.rknn'
MEMORY_LOG_FILE = 'memory_log.txt'

stop = False

tracemalloc.start()

def show_top(label='label') :
    with open(MEMORY_LOG_FILE, "w", encoding='utf-8') as f :
        f.write(f'Start! \n\n')
    while True:
        if stop :
            break
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        with open(MEMORY_LOG_FILE, "w", encoding='utf-8') as f :
            f.write(f'\n==== Momory at {label} ==== \n')
            for stat in top_stats[:25] :
                f.write(str(stat) + "\n")

        time.sleep(2)

def set_track(option_num, box_queue, track_queue, stop_queue, sort_frame_queue) :
    track = None
    match option_num :
        case 1:
            track = Sort(
                max_age=3,
                min_hits=3,
                iou_threshold=0.3
            )
            track.set_box_q(box_queue)
            track.set_track_q(track_queue)
            track.set_stop_q(stop_queue)
        case 2:
            track = Sort_addme(
                max_age=3,
                min_hits=3,
                iou_threshold=0.3
            )

            track_addme_cnn_model = get_model(SORT_CNN_MODEL_NAME, 0x2)
            track.set_model(track_addme_cnn_model)
            track.set_box_q(box_queue)
            track.set_track_q(track_queue)
            track.set_stop_q(stop_queue)
            track.set_sort_frame_q(sort_frame_queue)
        case _:
            print('no')
            return 1

    return track
def run(option_num) :
    global stop
    #log_writer
    logger = Logger.get_logger('npu_dSort')
    logger.info(f'Programe START! option = {option_num}')
    #camera, canvas, main - put, all - get
    stop_queue = queue.Queue(maxsize=1)
    #camera - put, yolo - get
    frame_queue = queue.Queue()
    #camera - put, sort_addme -get
    sort_frame_queue = queue.Queue()
    #yolo - put, track - get
    box_queue = queue.Queue()
    #track - put, canvas - get
    track_queue = queue.Queue()
    #camera - put, canvas - get 
    copy_frame_queue = queue.Queue()
    #camera - get, put, monitor - get
    fps_queue = queue.Queue()
    #canvas - put, main - get
    drawed_queue = queue.Queue()
    #main - get, yolo - put
    no_detect_flag = queue.Queue()

    logger.info('Queue create')

    #object setting
    camera = Cam()
    camera.set_stop_q(stop_queue)
    camera.set_cap(DEVICE_PORT)
    camera.set_frame_q(frame_queue)
    camera.set_frame_size(640, 640)
    camera.set_fps_q(fps_queue)
    camera.set_sort_frame_q(sort_frame_queue)
    camera.set_copy_frame_q(copy_frame_queue)

    yolo = Yolov5()
    yolo.set_model(MODEL_NAME)
    yolo.set_frame_q(frame_queue)
    yolo.set_box_q(box_queue)
    yolo.set_stop_q(stop_queue)
    yolo.set_no_detect_flag(no_detect_flag)

    canvas = Draw()
    canvas.set_frame_q(copy_frame_queue)
    canvas.set_track_q(track_queue)
    canvas.set_stop_q(stop_queue)
    canvas.set_draw_q(drawed_queue)
    canvas.set_no_detect_flag(no_detect_flag)

    track = None

    track= set_track(option_num, box_queue, track_queue, stop_queue, sort_frame_queue)

    if option_num == 1 :
        camera.set_sort_frame_q(True)

    if track == 1 :
        camera.set_sort_frame_q(True)
        return 1

    monitor = Monitor()
    monitor.set_stop_q(stop_queue)
    monitor.set_fps_q(fps_queue)

    #log_writer setting
    camera.set_logger(logger)
    yolo.set_logger(logger)
    canvas.set_logger(logger)
    monitor.set_logger(logger)

    #thread setting
    threads = []

    cam_thread = WatchThread(target=camera.run, name='cam')
    yolo_thread = WatchThread(target=yolo.run, name='yolo')
    track_thread = WatchThread(target=track.run, name='track')
    canvas_thread = WatchThread(target=canvas.run, name='canvas')
    monitor_thread = WatchThread(target=monitor.run, name='monitor')
    #memort rick, find
    #memory_thread = WatchThread(target=show_top, name='n')

    threads.append(cam_thread)
    threads.append(yolo_thread)
    threads.append(track_thread)
    threads.append(canvas_thread)
    threads.append(monitor_thread)
    #threads.append(memory_thread)

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

        frame_queue_size = frame_queue.qsize()
        box_queue_size = box_queue.qsize()
        track_queue_size = track_queue.qsize()
        copy_frame_queue_size = copy_frame_queue.qsize()
        drawed_queue_size = drawed_queue.qsize()

        q_size_sum = frame_queue_size + box_queue_size + track_queue_size + copy_frame_queue_size + drawed_queue_size

        if q_size_sum > 200 :
            stop_queue.put(True)
            emergency_stop = True
            stop = True
            logger.error('The sum of all queue size is 200 or more')
            logger.info('Emergency stop has been activated.')
            break

        monitor.update("queue_camera_in", frame_queue_size)
        monitor.update("queue_detect", box_queue_size)
        monitor.update("queue_track", track_queue_size)
        monitor.update("queue_capture", copy_frame_queue_size)

        try :
            output_img = drawed_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.001)
            continue
        cv2.imshow("show", output_img)

        input_key = cv2.waitKey(1) & 0xFF
        output_img = None

        if input_key == ord('q') :
            print("\n\n\033[1;31m⚠️  [EXIT] Termination command 'q' received. Stopping threads...\033[0m\n")
            logger.info("  [EXIT] Termination command 'q' received. Stopping threads... ")
            stop_queue.put(True)
            stop = True
            break
            
    cv2.destroyAllWindows()
    #thread close
    for i in range(threads_lenth):
        threads[i].thread_end()

    return emergency_stop

def parse_arg() :
    parser = argparse.ArgumentParser(description="Select sorting mode")
    
    parser.add_argument(
        "--option",
        type=str,
        choices=["sort", "sort_addme", "dSort"],
        default="sort",                           
        help="Choose sorting mode: sort (default), sort_addme, or dSort"
    )

    arg = parser.parse_args()
    return arg

option_map = {
    "sort": 1,
    "sort_addme": 2,
    "dSort": 3
}

def main() :
    arg = parse_arg()
    option_num = option_map[arg.option]
    is_run_fail = False
    while True :
        is_run_fail = run(option_num)
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