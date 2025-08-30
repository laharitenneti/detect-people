import cv2
import numpy as np

body_classifier = cv2.CascadeClassifier("data/haarcascade_upperbody.xml")

cap = cv2.VideoCapture("data/video.mov")

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(grey, 1.05, 5, minSize=(50, 50))

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
    cv2.imshow("People", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows
