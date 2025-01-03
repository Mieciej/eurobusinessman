import numpy as np
import cv2 as cv
import glob
import statistics

## PARAMETERS
DOT_THRESHOLD  = 0.66 # (0, 1.0)
N_DICE_RESULT_VOTES = 101 # How many results are used to determine dice result




dice_template = cv.imread("dice.png", cv.IMREAD_GRAYSCALE)
dot_template = cv.imread("dot.png", cv.IMREAD_GRAYSCALE)
cap = cv.VideoCapture("output0.mp4")

w = 32
h = 32

dot_w = 5
dot_h = 5

roll_votes = [0] * N_DICE_RESULT_VOTES
roll_votes_idx = 0
roll_result = -1
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
    cv.rectangle(rescaled, top_left, bottom_right, 255, 1)

    dice_img = gray[ top_left[1] : top_left[1] + h, top_left[0]:top_left[0] + w]

    match = cv.matchTemplate(dice_img, dot_template, cv.TM_CCOEFF_NORMED)
    loc = np.where( match >= DOT_THRESHOLD)
    dots = []
    for pt in zip(*loc[::-1]):
        unique = True
        for dot in dots:
            if np.sqrt((dot[0] - pt[0])**2 + (dot[1] - pt[1])**2) < 2:
                unique = False
                break
        if unique:
            cv.rectangle(dice_img, pt, (pt[0] + dot_w, pt[1] + dot_h), 255, 1)
            dots.append(pt)

    roll_vote = len(dots)
    if roll_vote >= 1 and roll_vote <= 6:
        roll_votes[roll_votes_idx] = roll_vote
        roll_votes_idx = (roll_votes_idx + 1) % N_DICE_RESULT_VOTES
        roll_result = statistics.mode(roll_votes)

    if roll_result > 1:
        font = cv.FONT_HERSHEY_PLAIN
        bottomLeftCornerOfText = (25, rescaled.shape[0] - 50)
        fontScale = 2
        fontColor = (255,255,255)
        thickness = 1
        lineType = 2

        cv.putText(rescaled,f"Roll Result: {roll_result}",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)


    cv.imshow('frame', rescaled)
    cv.imshow('dice', dice_img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
