Environment

Device: Orange Pi 5 Pro

Camera: OV13850 (using MIPI port)
(
The camera only runs at 15 FPS, even though according to https://ae01.alicdn.com/kf/S27fdfd5331a64c5f83b887cb40da1e63f.jpg
 it should support up to 30 FPS.
Other users on https://github.com/orangepi-xunlong/orangepi-build/issues/185
 are also experiencing the same 15 FPS limitation.
If you know a possible solution, please contact me
)

Installation
pip install -r requirements.txt

Usage
python main.py

Features

Capture video from OV13850 camera.

Run YOLOv5s model on NPU using RKNN.
