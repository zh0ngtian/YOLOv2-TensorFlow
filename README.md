Introduction<br>
===
This is a repo for implementing YOLOv2 with trained models on tensorflow.<br>

How to use<br>
===
* Download the weights file [here](https://1drv.ms/u/s!AkKw30iZFJzBhXWfdbCEMmadqeIG) (one drive). The weights file includes 3 files, they should be put in `weights` folder.<br>
* The images should be put in `test_pic` folder, and the video should be put in `test_video` folder.<br>
* Then Edit the configure file `cfg.py`, or use the default configure.<br>
* Use `python YOLO_coco_test.py` to run.<br>

Demo<br>
===
![demo](https://github.com/YaoZhongtian/YOLOv2-test-with-TensorFlow/raw/master/demo/demo_1.gif)<br>
![demo](https://github.com/YaoZhongtian/YOLOv2-test-with-TensorFlow/raw/master/demo/demo.jpg)<br>

Reference<br>
===
[pjreddie/darknet](https://github.com/pjreddie/darknet)<br>
[gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)<br>

Requirements<br>
===
* TensorFlow<br>
* OpenCV
