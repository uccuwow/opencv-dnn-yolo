import argparse
import os
from cv2 import cv2
import time

# model_class = "model/coco.names"
# model_weight = "model/yolov3-tiny.weights"
# model_cfg = "model/yolov3-tiny.cfg"
# input_file = "data/demo.mp4"

# CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--weights", default="model/yolov3-tiny.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./model/yolov3-tiny.weights",
                        help="path to config file")
    parser.add_argument("--class_file", default="./model/coco.names",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")

    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.class_file):
        raise(ValueError("Invalid class file path {}".format(os.path.abspath(args.class_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid input file path {}".format(os.path.abspath(args.input))))

def check_type(images_path):
    """
    Determine input file is video, image or others.
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return 0
    elif input_path_extension in ['mp4', 'avi', 'webm']:
        return 1
    else:
        return 2

def detect_img(input_file, cfg, weights, classfile, thresh, dont_show, obj_infos):
    class_names = []
    with open(classfile, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    net = cv2.dnn.readNet(weights, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    
    frame = cv2.imread(input_file)

    start = time.time()
    classes, scores, boxes = model.detect(frame, thresh, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        if(obj_infos):
            label += " (%f, %f, %f, %f)" % (box[0], box[1], box[2], box[3]) 
        print(label)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    print(fps_label + "\r",end="", flush = True)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if(not dont_show):
        while cv2.waitKey(1) < 1:
            cv2.imshow("video detections", frame)

    cv2.destroyAllWindows()

def detect_video(input_file, cfg, weights, classfile, thresh, dont_show, obj_infos):
    class_names = []
    with open(classfile, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    vc = cv2.VideoCapture(input_file)

    net = cv2.dnn.readNet(weights, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    while cv2.waitKey(1) < 1:
        (grabbed, frame) = vc.read()
        if not grabbed:
            exit()

        start = time.time()
        classes, scores, boxes = model.detect(frame, thresh, NMS_THRESHOLD)
        end = time.time()

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            if(obj_infos):
                label += " (%f, %f, %f, %f)" % (box[0], box[1], box[2], box[3])
            print(label)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()
        
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        print(fps_label + "\r",end="", flush = True)
        
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if(not dont_show):
            cv2.imshow("video detections", frame)
        
    vc.release()
    cv2.destroyAllWindows()

def main():
    args = parser()
    check_arguments_errors(args)

    thresh = args.thresh
    dont_show = args.dont_show
    obj_info = args.ext_output

    file_type = check_type(args.input)
    if (file_type == 0):
        detect_img(args.input, args.config_file, args.weights, args.class_file, thresh, dont_show, obj_info)
    elif file_type == 1:
        detect_video(args.input, args.config_file, args.weights, args.class_file, thresh, dont_show, obj_info)


if __name__ == "__main__":
    main()
