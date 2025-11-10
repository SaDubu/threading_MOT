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

DEVICE_PORT = "/dev/video11"
MODEL_NAME = "yolo.rknn"

def main() :
    stop_queue = queue.Queue()
    frame_queue = queue.Queue()
    box_queue = queue.Queue()
    track_queue = queue.Queue()
    copy_frame_queue = queue.Queue()

    #object setting
    camera = Cam()
    camera.set_stop_q(stop_queue)
    camera.set_cap(DEVICE_PORT)
    camera.set_frame_q(frame_queue)
    camera.set_frame_size(640, 640)

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

    track = Sort(
        max_age=3,
        min_hits=1,
        iou_threshold=0.3
    )
    track.set_box_q(box_queue)
    track.set_track_q(track_queue)
    track.set_stop_q(stop_queue)



    #thread setting
    cam_thread = WatchThread(target=camera.run, name='cam')
    yolo_thread = WatchThread(target=yolo.run, name='yolo')
    track_thread = WatchThread(target=track.run, name='track')
    canvas_thread = WatchThread(target=canvas.run, name='canvas')

    #test
    frame_queue_size = 0
    box_queue_size = 0
    track_queue_size = 0
    copy_frame_queue_size = 0
    
    q_size_sum = 0
    count = 0

    while True:
        if stop_queue.qsize() != 0:
            break

        cam_thread.restart_if_out()
        yolo_thread.restart_if_out()
        track_thread.restart_if_out()
        canvas_thread.restart_if_out()

        frame_queue_size = frame_queue.qsize()
        box_queue_size = box_queue.qsize()
        track_queue_size = track_queue.qsize()
        copy_frame_queue_size = copy_frame_queue.qsize()

        print(f'\rframe_q size = {frame_queue_size}, box_q size = {box_queue_size}, trk_q size = {track_queue_size} copy_frame_q_size = {copy_frame_queue_size}', end='')
        sys.stdout.write("\r" + "" *50 +"\r")
        sys.stdout.flush()

        # 🔹 키보드 입력 체크 (0.1초마다)
        dr, _, _ = select.select([sys.stdin], [], [], 0.1)
        if dr:
            cmd = sys.stdin.readline().strip()
            input_key = cmd.lower()
            if input_key == 'q':
                print("\n\033[1;31m⚠️  [EXIT] Termination command 'q' received. Stopping threads...\033[0m\n")
                stop_queue.put(True)
                break
            

    #thread close
    cam_thread.thread_end()
    yolo_thread.thread_end()
    track_thread.thread_end()
    canvas_thread.thread_end()

    return 0

if __name__ == "__main__" :
    main()