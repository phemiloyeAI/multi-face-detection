# A Crowd Counting Application
### About 
This app uses a lightweight face-detector to detect regions containing faces from image and video streams and give a total count of the people present.
It combines face-tracking with the face detector to give an accurate count of the people present in a video stream.

### Run Demo
- Clone repo.
- Install requiremets
```
pip install -r requiremnts.txt
```
- Run the following command:
```
python face_detection.py --weights weights\weights.pt --input demo\selfie.jpg --output results.jpg
```
### Result
![Image](demo\image.jpg)

### References
[https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

***paper***
```
@article{YOLO5Face,
title = {YOLO5Face: Why Reinventing a Face Detector},
author = {Delong Qi and Weijun Tan and Qi Yao and Jingfeng Liu},
booktitle = {ArXiv preprint ArXiv:2105.12931},
year = {2021}
}