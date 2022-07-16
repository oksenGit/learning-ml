import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ml\\rush\\FaceRecognition\\attendance_images'

images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}\\{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceEncoding = face_recognition.face_encodings(img)[0]
        encodings.append(faceEncoding)
    return encodings


def markAttendance(name):
    with open('ml\\rush\\FaceRecognition\\Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)

print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imsSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imsSmall)
    encodingsCurFrame = face_recognition.face_encodings(
        imsSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            markAttendance(name)
            y1, x2, y2, x1 = faceLoc
            y1 *= 4
            x2 *= 4
            y2 *= 4
            x1 *= 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 + 35), (x2+5, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name+" "+str(round(faceDist[matchIndex], 2)),
                        (x1 + 6, y2 + 27), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
