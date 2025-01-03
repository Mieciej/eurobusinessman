import numpy as np
import cv2 as cv


cap = cv.VideoCapture("output0.mp4")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=0.3, fy=0.3, interpolation = cv.INTER_CUBIC)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
