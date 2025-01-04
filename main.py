import numpy as np
import cv2 as cv
import glob
import statistics
from dataclasses import dataclass

## PARAMETERS
DOT_THRESHOLD  = 0.66 # (0, 1.0)
N_DICE_RESULT_VOTES = 101 # How many results are used to determine dice result

@dataclass
class Template:
    img: np.ndarray
    width: int
    height: int

dice_template = Template(cv.imread("dice.png", cv.IMREAD_GRAYSCALE), 32, 32)
dot_template = Template(cv.imread("dot.png", cv.IMREAD_GRAYSCALE), 5, 5)
pawn_template = Template(cv.imread("pawn.png", cv.IMREAD_GRAYSCALE), 32, 32)
board_template = Template(cv.imread("board.png", cv.IMREAD_GRAYSCALE), 620, 620)

roll_votes = [0] * N_DICE_RESULT_VOTES
roll_votes_idx = 0
roll_result = -1


def get_field(pos):
    x, y = pos
    if 17 <= x and 532 <= y and x <= 97  and y <= 611:
            return "1. START"
    elif 16 <= x and 486 <= y and x <= 78  and y <= 526:
            return "2. SALONIKI"
    elif 15 <= x and 439 <= y and x <= 97  and y <= 480:
            return "3. NIEBIESKA SZANSA"
    elif 14 <= x and 392 <= y and x <= 77  and y <= 433:
            return "4. ATENY"
    elif 13 <= x and 345 <= y and x <= 95  and y <= 387:
            return "5. STRZEŻONY PARKING"
    elif 11 <= x and 298 <= y and x <= 94  and y <= 340:
            return "6. KOLEJE POŁUDNIOWE"
    elif 10 <= x and 250 <= y and x <= 75  and y <= 292:
            return "7. NEAPOL"
    elif 10 <= x and 203 <= y and x <= 93  and y <= 245:
            return "8. CZERWONA SZANSA"
    elif 7 <= x and 154 <= y and x <= 72  and y <= 196:
            return "9. MEDIOLAN"
    elif 6 <= x and 105 <= y and x <= 71  and y <= 148:
            return "10. MEDIOLAN"
    elif 4 <= x and 14 <= y and x <= 90  and y <= 100:
            return "11. WIĘZIENIE"
    elif 93 <= x and 14 <= y and x <= 137  and y <= 79:
            return "12. BARCELONA"
    elif 142 <= x and 13 <= y and x <= 186  and y <= 99:
            return "13. ELEKTROWNIA"
    elif 190 <= x and 13 <= y and x <= 234  and y <= 79:
            return "14. SEWILLA"
    elif 238 <= x and 13 <= y and x <= 282  and y <= 79:
            return "15. MADRYT"
    elif 288 <= x and 13 <= y and x <= 330  and y <= 98:
            return "16. KOLEJE ZACHODNIE"
    elif 336 <= x and 12 <= y and x <= 379  and y <= 78:
            return "17. LIVERPOLL"
    elif 385 <= x and 13 <= y and x <= 427  and y <= 98:
            return "18. NIEBIESKA SZANSA"
    elif 434 <= x and 12 <= y and x <= 476  and y <= 78:
            return "19. GLASGOW"
    elif 482 <= x and 12 <= y and x <= 524  and y <= 78:
            return "20. LONDYN"
    elif 531 <= x and 11 <= y and x <= 613  and y <= 97:
            return "21. DARMOWY PARKING"
    elif 549 <= x and 103 <= y and x <= 612  and y <= 146:
            return "22. ROTTERDAM"
    elif 549 <= x and 153 <= y and x <= 611  and y <= 195:
            return "23. CZERWONA SZANSA"
    elif 548 <= x and 201 <= y and x <= 610  and y <= 243:
            return "24. BRUKSELA"
    elif 545 <= x and 249 <= y and x <= 608  and y <= 291:
            return "25. AMSTERDAM"
    elif 526 <= x and 298 <= y and x <= 608  and y <= 339:
            return "26. KOLEJE POLNOCNE"
    elif 545 <= x and 345 <= y and x <= 607  and y <= 386:
            return "27. MALMO"
    elif 544 <= x and 392 <= y and x <= 606  and y <= 433:
            return "28. GOTEBORG"
    elif 524 <= x and 439 <= y and x <= 605  and y <= 480:
            return "29. SIEC WODOCIAGOWA"
    elif 543 <= x and 487 <= y and x <= 605  and y <= 527:
            return "30. SZTOKHOLM"
    elif 523 <= x and 532 <= y and x <= 603  and y <= 612:
            return "31. IDZ DO WIEZIENIA"
    elif 476 <= x and 550 <= y and x <= 517  and y <= 611:
            return "32. FRANKFURT"
    elif 429 <= x and 550 <= y and x <= 470  and y <= 612:
            return "33. KOLONIA"
    elif 382 <= x and 531 <= y and x <= 423  and y <= 611:
            return "34. NIEBIESKA SZANSA"
    elif 336 <= x and 550 <= y and x <= 377  and y <= 611:
            return "35. BONN"
    elif 289 <= x and 531 <= y and x <= 330  and y <= 612:
            return "36. KOLEJE WSCHODNIE"
    elif 242 <= x and 550 <= y and x <= 284  and y <= 611:
            return "37. CZERWONA SZANSA"
    elif 195 <= x and 550 <= y and x <= 237  and y <= 610:
            return "38. INNSBRUCK"
    elif 149 <= x and 531 <= y and x <= 191  and y <= 611:
            return "39. PODATEK"
    elif 102 <= x and 550 <= y and x <= 144  and y <= 611:
            return "40. WIEDEN"
    else:
        return "UKNOWN"

cap = cv.VideoCapture("output0.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    rescaled = cv.resize(frame, None, fx=0.3, fy=0.3, interpolation = cv.INTER_CUBIC)
    gray = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)

    match = cv.matchTemplate(gray, board_template.img, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
    board_top_left = max_loc
    board_img = rescaled[ board_top_left[1] : board_top_left[1] + board_template.height, board_top_left[0]:board_top_left[0] + board_template.width]

    gray_board = cv.cvtColor(board_img, cv.COLOR_BGR2GRAY)
    match = cv.matchTemplate(gray_board, dice_template.img, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0] + dice_template.width, top_left[1] + dice_template.height)
    cv.rectangle(board_img, top_left, bottom_right, 255, 1)

    dice_img = gray_board[ top_left[1] : top_left[1] + dice_template.height, top_left[0]:top_left[0] + dice_template.width]
    match = cv.matchTemplate(dice_img, dot_template.img, cv.TM_CCOEFF_NORMED)
    loc = np.where( match >= DOT_THRESHOLD)
    dots = []
    for pt in zip(*loc[::-1]):
        unique = True
        for dot in dots:
            if np.sqrt((dot[0] - pt[0])**2 + (dot[1] - pt[1])**2) < 2:
                unique = False
                break
        if unique:
            cv.rectangle(dice_img, pt, (pt[0] + dot_template.width, pt[1] + dot_template.height), 255, 1)
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

    match = cv.matchTemplate(gray_board, pawn_template.img, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
    pawn_top_left = max_loc
    bottom_right = (pawn_top_left[0] + pawn_template.width, pawn_top_left[1] + pawn_template.height)
    cv.rectangle(board_img, pawn_top_left, bottom_right, 255, 1)
    if max_val > 0.92:
        font = cv.FONT_HERSHEY_PLAIN
        bottomLeftCornerOfText = (25, rescaled.shape[0] - 80)
        fontScale = 2
        fontColor = (255,255,255)
        thickness = 1
        lineType = 2
        cv.putText(rescaled,f"Bue Pawn Pos: {get_field((pawn_top_left[0] + pawn_template.width/2,pawn_top_left[1] + pawn_template.height/2))}",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

    cv.imshow('frame', rescaled)
    cv.imshow('dice', dice_img)
    cv.imshow('board', board_img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



