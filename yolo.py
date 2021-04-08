import cv2 as cv
import numpy as np

model_bin = "model/yolov3-tiny.weights"
config_text = "model/yolov3-tiny.cfg"

# Load names of classes
classes = None
with open("model/coco.names", 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load tensorflow model
net = cv.dnn.readNetFromDarknet(config_text, model_bin)
image = cv.imread("data/giraffe.jpg")
h = image.shape[0]
w = image.shape[1]

# 获得所有层名称与索引
layerNames = net.getLayerNames()
lastLayerId = net.getLayerId(layerNames[-1])
lastLayer = net.getLayer(lastLayerId)
print(lastLayer.type)

# 基于多个Region层输出getUnconnectedOutLayersNames
blobImage = cv.dnn.blobFromImage(image, 1.0/255.0, (416, 416), None, True, False);
outNames = net.getUnconnectedOutLayersNames()
net.setInput(blobImage)
outs = net.forward(outNames)

# Put efficiency information.
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

print(outs)

# 绘制检测矩形
classIds = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        # numbers are [center_x, center_y, width, height]
        if confidence > 0.5:
            center_x = int(detection[0] * w)
            center_y = int(detection[1] * h)
            width = int(detection[2] * w)
            height = int(detection[3] * h)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            classIds.append(classId)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in indices:
    i = i[0]
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    cv.rectangle(image, (left, top), (left+width, top+height), (0, 0, 255), 2, 8, 0)
    cv.putText(image, classes[classIds[i]], (left, top), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

cv.imshow('YOLOv3-Detection-Demo', image)

cv.waitKey(0)
cv.destroyAllWindows()
