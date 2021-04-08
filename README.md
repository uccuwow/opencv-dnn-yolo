# opencv-dnn-yolo

## Environment
* OpenCV version >= 4.4

## Build C++ code
```
g++ -std=c++11 yolov4_opencv_dnn_my.cpp \
-I /where/did/i/install/opencv/include \
-L /where/did/i/install/opencv/lib -o dnn_c
```

## Build Python code
```
python3 yolo.py
# or 
python3 yolov4.py
```
