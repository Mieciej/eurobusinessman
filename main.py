import numpy as np
import cv2 as cv

dice_template = cv.imread('dice.png', cv.IMREAD_GRAYSCALE)


cap = cv.VideoCapture("output0.mp4")

w, h = dice_template.shape[::-1]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    rescaled = cv.resize(frame, None, fx=0.3, fy=0.3, interpolation = cv.INTER_CUBIC)
    gray = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    match = cv.matchTemplate(gray, dice_template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(rescaled, top_left, bottom_right, 255, 2)
    x = max_loc[1]
    y = max_loc[0]
    area = 10
    match[x-area:x+area, y-area:y+area] = 0
    min_val, max_val2, min_loc, max_loc = cv.minMaxLoc(match)
    print(max_val, max_val2)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(rescaled, top_left, bottom_right, 255, 2)

    # loc = np.where(match >= 0.6)
    # for pt in zip (*loc[::-1]):
        # cv.rectangle(rescaled, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    print(np.max(match))
    cv.imshow('frame', rescaled)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
