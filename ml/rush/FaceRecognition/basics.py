import cv2
import numpy as np
import face_recognition


def loadAndConvertImage(path):
    img = face_recognition.load_image_file(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def makeFaceRectangle(img):
    faceLoc = face_recognition.face_locations(img)[0]
    cv2.rectangle(img, (faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]), (255, 0, 0), 2)
  
def getFaceEncoding(img):
    faceEncoding = face_recognition.face_encodings(img)[0]
    return faceEncoding

imgElon = loadAndConvertImage(
    "python\\rush\\FaceRecognition\\images\\elon_clear.jpg")
imgTest = loadAndConvertImage(
    "python\\rush\\FaceRecognition\\images\\elon_test.jpeg")

makeFaceRectangle(imgElon)
elonEncoding = getFaceEncoding(imgElon)

makeFaceRectangle(imgTest)
testEncoding = getFaceEncoding(imgTest)


results = face_recognition.compare_faces([elonEncoding], testEncoding)

faceDistance = face_recognition.face_distance([elonEncoding], testEncoding)

print(results, faceDistance)
cv2.putText(imgTest, f'{results[0]} {round(faceDistance[0],2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Elon Musk", imgElon)
cv2.imshow("Test", imgTest)

cv2.waitKey(0)
