import cv2

path = 'ml\\rush\\ObjectRecognition'
configPath = f'{path}\\obj_det\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = f'{path}\\obj_det\\frozen_inference_graph.pb'


#img = cv2.imread(f'{path}\\lena.png')

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
with open(f'{path}\\obj_det\\coco.names', 'r') as f:
    for line in f:
        classNames.append(line.strip())

net = cv2.dnn_DetectionModel(configPath, weightsPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confidences, bbox = net.detect(img, confThreshold=0.5)

    if(len(classIds) != 0):
        for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(1)
