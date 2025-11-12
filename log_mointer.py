# system_monitor.py
import psutil
import time
# logger.py
import logging
import os
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = "log"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "level": record.levelname,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "module": record.module,
            "message": record.getMessage(),
        }
        return json.dumps(log, ensure_ascii=False)

class Logger :
    def __init__(self) :
        self.text = text

    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # --- System Log Handler ---
        system_handler = RotatingFileHandler(
            f"{LOG_DIR}/system.log", maxBytes=5*1024*1024, backupCount=10
        )
        system_handler.setFormatter(JsonFormatter())
        system_handler.setLevel(logging.INFO)

        # --- Error Log Handler ---
        error_handler = RotatingFileHandler(
            f"{LOG_DIR}/error.log", maxBytes=5*1024*1024, backupCount=10
        )
        error_handler.setFormatter(JsonFormatter())
        error_handler.setLevel(logging.ERROR)

        if not logger.handlers:
            logger.addHandler(system_handler)
            logger.addHandler(error_handler)

        return logger

class Monitor:
    def __init__(self, interval=1):
        self.interval = interval

        self.stop_q = None
        self.past_frame_lenth = 0
        self.fps_q = None

        # External data (FPS, Queue sizes...)
        self.data = {
            "camera_fps": 0,
            "queue_capture": 0,
            "queue_detect": 0,
            "queue_track": 0,
            "queue_camera_in": 0
        }
    
    def set_stop_q(self, q) :
        self.stop_q = q

    def set_fps_q(self, q) :
        self.fps_q = q

    def calculate_fps(self) :
        lenth = self.fps_q.get()
        self.past_frame_lenth = lenth - self.past_frame_lenth
        fps = self.past_frame_lenth / self.interval
        self.past_frame_lenth = lenth
        self.update("camera_fps", fps)

    def update(self, key, value):
        self.data[key] = value

    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return int(f.read()) / 1000
        except:
            return -1

    def is_stop(self) :
        stop_q_size = self.stop_q.qsize()
        if stop_q_size == 0 :
            return False
        return True

    def run(self):
        while True:
            self.calculate_fps()
            if self.is_stop() :
                break
            log_data = {
                "fps": {
                    "camera": self.data["camera_fps"],
                },
                "system": {
                    "cpu_usage": psutil.cpu_percent(),
                    "cpu_temp": self.get_cpu_temp(),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
                },
                "queues": {
                    "camera_in" : self.data["queue_camera_in"],
                    "capture": self.data["queue_capture"],
                    "detect": self.data["queue_detect"],
                    "track": self.data["queue_track"],
                }
            }

            terminal_msg = (
                f"[FPS] cam:{self.data['camera_fps']:.1f} | "
                f"[CPU] {log_data['system']['cpu_usage']:.0f}% {log_data['system']['cpu_temp']:.1f}°C | "
                f"[MEM] {log_data['system']['memory_mb']:.0f}MB | "
                f"[Q] C:{log_data['queues']['capture']} "
                f"D:{log_data['queues']['detect']} "
                f"T:{log_data['queues']['track']} "
                f"C_i:{log_data['queues']['camera_in']}"
            )

            print("\r" + terminal_msg + "    ", end="", flush=True)

            time.sleep(self.interval)

    def __del__(self):
        print('\nmonitor class destructor')