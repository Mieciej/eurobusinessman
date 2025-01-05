import numpy as np
import cv2 as cv
import glob
import statistics
from dataclasses import dataclass

## PARAMETERS
N_PAWN_VOTES = 81
PAWN_THRESHOLD = 2000000

@dataclass
class Template:
    img: np.ndarray
    width: int
    height: int


blue_pawn_template = Template(cv.imread("blue_pawn.png", cv.IMREAD_GRAYSCALE), 32, 32)
red_pawn_template = Template(cv.imread("red_pawn.png", cv.IMREAD_GRAYSCALE), 32, 32)
green_pawn_template = Template(cv.imread("green_pawn.png", cv.IMREAD_GRAYSCALE), 32, 32)

blue_pawn_hard_template = Template(cv.imread("blue_pawn_hard.png", cv.IMREAD_GRAYSCALE), 32, 32)
green_pawn_hard_template = Template(cv.imread("green_pawn_hard.png", cv.IMREAD_GRAYSCALE), 32, 32)
blue_pawn_hard_close_template = Template(cv.imread("blue_pawn_hard_close.png", cv.IMREAD_GRAYSCALE), 32, 32)
green_pawn_hard_close_template = Template(cv.imread("green_pawn_hard_close.png", cv.IMREAD_GRAYSCALE), 32, 32)

pawn_templates = None
board_normal_template = Template(cv.imread("board.png", cv.IMREAD_GRAYSCALE), 620, 620)
board_hard_template = Template(cv.imread("board_hard.png", cv.IMREAD_GRAYSCALE), 604, 843)


pawn_pos_votes = [[-1] * N_PAWN_VOTES, [-1] * N_PAWN_VOTES, [-1] * N_PAWN_VOTES]
pawn_votes_idx = [0, 0, 0]
pawn_pos = [-1, -1, -1]
prev_pawn_pos = [-1, -1, -1]

field_names = [ "1. START",
                "2. SALONIKI",
                "3. NIEBIESKA SZANSA",
                "4. ATENY",
                "5. STRZEZONY PARKING",
                "6. KOLEJE POLUDNIOWE",
                "7. NEAPOL",
                "8. CZERWONA SZANSA",
                "9. MEDIOLAN",
                "10. RZYM",
                "11. WIEZIENIE",
                "12. BARCELONA",
                "13. ELEKTROWNIA",
                "14. SEWILLA",
                "15. MADRYT",
                "16. KOLEJE ZACHODNIE",
                "17. LIVERPOLL",
                "18. NIEBIESKA SZANSA",
                "19. GLASGOW",
                "20. LONDYN",
                "21. DARMOWY PARKING",
                "22. ROTTERDAM",
                "23. CZERWONA SZANSA",
                "24. BRUKSELA",
                "25. AMSTERDAM",
                "26. KOLEJE POLNOCNE",
                "27. MALMO",
                "28. GOTEBORG",
                "29. SIEC WODOCIAGOWA",
                "30. SZTOKHOLM",
                "31. IDZ DO WIEZIENIA",
                "32. FRANKFURT",
                "33. KOLONIA",
                "34. NIEBIESKA SZANSA",
                "35. BONN",
                "36. KOLEJE WSCHODNIE",
                "37. CZERWONA SZANSA",
                "38. INNSBRUCK",
                "39. PODATEK",
                "40. WIEDEN"
        ]


def get_field_normal(pos):
    x, y = pos
    if 17 <= x and 532 <= y and x <= 97  and y <= 611:
        return 0
    elif 16 <= x and 486 <= y and x <= 78  and y <= 526:
        return 1
    elif 15 <= x and 439 <= y and x <= 97  and y <= 480:
        return 2
    elif 14 <= x and 392 <= y and x <= 77  and y <= 433:
        return 3
    elif 13 <= x and 345 <= y and x <= 95  and y <= 387:
        return 4
    elif 11 <= x and 298 <= y and x <= 94  and y <= 340:
        return 5
    elif 10 <= x and 250 <= y and x <= 75  and y <= 292:
        return 6
    elif 10 <= x and 203 <= y and x <= 93  and y <= 245:
        return 7
    elif 7 <= x and 154 <= y and x <= 72  and y <= 196:
        return 8
    elif 6 <= x and 105 <= y and x <= 71  and y <= 148:
        return 9
    elif 4 <= x and 14 <= y and x <= 90  and y <= 100:
        return 10
    elif 93 <= x and 14 <= y and x <= 137  and y <= 79:
        return 11
    elif 142 <= x and 13 <= y and x <= 186  and y <= 99:
        return 12
    elif 190 <= x and 13 <= y and x <= 234  and y <= 79:
        return 13
    elif 238 <= x and 13 <= y and x <= 282  and y <= 79:
        return 14
    elif 288 <= x and 13 <= y and x <= 330  and y <= 98:
        return 15
    elif 336 <= x and 12 <= y and x <= 379  and y <= 78:
        return 16
    elif 385 <= x and 13 <= y and x <= 427  and y <= 98:
        return 17
    elif 434 <= x and 12 <= y and x <= 476  and y <= 78:
        return 18
    elif 482 <= x and 12 <= y and x <= 524  and y <= 78:
        return 19
    elif 531 <= x and 11 <= y and x <= 613  and y <= 97:
        return 20
    elif 549 <= x and 103 <= y and x <= 612  and y <= 146:
        return 21
    elif 549 <= x and 153 <= y and x <= 611  and y <= 195:
        return 22
    elif 548 <= x and 201 <= y and x <= 610  and y <= 243:
        return 23
    elif 545 <= x and 249 <= y and x <= 608  and y <= 291:
        return 24
    elif 526 <= x and 298 <= y and x <= 608  and y <= 339:
        return 25
    elif 545 <= x and 345 <= y and x <= 607  and y <= 386:
        return 26
    elif 544 <= x and 392 <= y and x <= 606  and y <= 433:
        return 27
    elif 524 <= x and 439 <= y and x <= 605  and y <= 480:
        return 28
    elif 543 <= x and 487 <= y and x <= 605  and y <= 527:
        return 29
    elif 523 <= x and 532 <= y and x <= 603  and y <= 612:
        return 30
    elif 476 <= x and 550 <= y and x <= 517  and y <= 611:
        return 31
    elif 429 <= x and 550 <= y and x <= 470  and y <= 612:
        return 32
    elif 382 <= x and 531 <= y and x <= 423  and y <= 611:
        return 33
    elif 336 <= x and 550 <= y and x <= 377  and y <= 611:
        return 34
    elif 289 <= x and 531 <= y and x <= 330  and y <= 612:
        return 35
    elif 242 <= x and 550 <= y and x <= 284  and y <= 611:
        return 36
    elif 195 <= x and 550 <= y and x <= 237  and y <= 610:
        return 37
    elif 149 <= x and 531 <= y and x <= 191  and y <= 611:
        return 38
    elif 102 <= x and 550 <= y and x <= 144  and y <= 611:
        return 39
    else:
        return -1

def get_hard_field(pos):
    x, y = pos
    if 508 <= x and 118 <= y and x <= 557 and y <= 185:
        return 0
    elif 520 <= x and 197 <= y and x <= 557 and y <= 240:
        return 1
    elif 509 <= x and 239 <= y and x <= 558 and y <= 281:
        return 2
    elif 522 <= x and 284 <= y and x <= 559 and y <= 322:
        return 3
    elif 510 <= x and 327 <= y and x <= 560 and y <= 364:
        return 4
    elif 510 <= x and 370 <= y and x <= 560 and y <= 406:
        return 5
    elif 524 <= x and 413 <= y and x <= 560 and y <= 448:
        return 6
    elif 512 <= x and 458 <= y and x <= 560 and y <= 490:
        return 7
    elif 525 <= x and 500 <= y and x <= 562 and y <= 531:
        return 8
    elif 525 <= x and 544 <= y and x <= 563 and y <= 574:
        return 9
    elif 512 <= x and 591 <= y and x <= 564 and y <= 653:
        return 10
    elif 480 <= x and 618 <= y and x <= 510 and y <= 672:
        return 11
    elif 448 <= x and 608 <= y and x <= 478 and y <= 683:
        return 12
    elif 411 <= x and 637 <= y and x <= 444 and y <= 693:
        return 13
    elif 372 <= x and 648 <= y and x <= 409 and y <= 705:
        return 14
    elif 331 <= x and 638 <= y and x <= 368 and y <= 719:
        return 15
    elif 286 <= x and 671 <= y and x <= 326 and y <= 732:
        return 16
    elif 239 <= x and 660 <= y and x <= 281 and y <= 748:
        return 17
    elif 185 <= x and 712 <= y and x <= 232 and y <= 764:
        return 18
    elif 127 <= x and 712 <= y and x <= 179 and y <= 781:
        return 19
    elif 9 <= x and 716 <= y and x <= 121 and y <= 799:
        return 20
    elif 9 <= x and 648 <= y and x <= 98 and y <= 689:
        return 21
    elif 10 <= x and 580 <= y and x <= 123 and y <= 621:
        return 22
    elif 12 <= x and 580 <= y and x <= 98 and y <= 561:
        return 23
    elif 13 <= x and 447 <= y and x <= 101 and y <= 498:
        return 24
    elif 14 <= x and 384 <= y and x <= 126 and y <= 435:
        return 25
    elif 15 <= x and 316 <= y and x <= 103 and y <= 374:
        return 26
    elif 17 <= x and 251 <= y and x <= 105 and y <= 313:
        return 27
    elif 19 <= x and 188 <= y and x <= 128 and y <= 253:
        return 28
    elif 22 <= x and 123 <= y and x <= 105 and y <= 191:
        return 29
    elif 22 <= x and 8 <= y and x <= 129 and y <= 134:
        return 30
    elif 136 <= x and 34 <= y and x <= 184 and y <= 119:
        return 31
    elif 191 <= x and 46 <= y and x <= 235 and y <= 128:
        return 32
    elif 241 <= x and 57 <= y and x <= 282 and y <= 158:
        return 33
    elif 288 <= x and 68 <= y and x <= 326 and y <= 144:
        return 34
    elif 331 <= x and 77 <= y and x <= 367 and y <= 170:
        return 35
    elif 372 <= x and 88 <= y and x <= 405 and y <= 176:
        return 36
    elif 409 <= x and 95 <= y and x <= 441 and y <= 163:
        return 37
    elif 444 <= x and 103 <= y and x <= 474 and y <= 186:
        return 38
    elif 476 <= x and 110 <= y and x <= 506 and y <= 174:
        return 39

get_field = None

cap = cv.VideoCapture("scaled_hard_output6.mp4")

events = []


player_names = [ "blue",
                 "red",
                 "green"
        ]

elapsed_frames = 0

board_template = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    elapsed_frames +=1
    # rescaled = cv.resize(frame, None, fx=0.3, fy=0.3, interpolation = cv.INTER_CUBIC)
    rescaled = frame
    gray = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)

    if not board_template:
        match = cv.matchTemplate(gray, board_normal_template.img, cv.TM_SQDIFF)
        min_val_normal, max_val_normal, min_loc_normal, max_loc_normal = cv.minMaxLoc(match)
        match = cv.matchTemplate(gray, board_hard_template.img, cv.TM_SQDIFF)
        min_val_hard, max_val_hard, min_loc_hard, max_loc_hard = cv.minMaxLoc(match)
        if min_val_normal < min_val_hard:
            board_template = board_normal_template
            pawn_templates = [[blue_pawn_template], [red_pawn_template], [green_pawn_template]]
            get_field = get_field_normal
            print("The board is easy")
        else:
            board_template = board_hard_template
            pawn_templates = [[blue_pawn_hard_template, blue_pawn_hard_close_template], [red_pawn_template], [green_pawn_hard_template, green_pawn_hard_close_template] ]
            get_field = get_hard_field
            print("The board is hard!")

    match = cv.matchTemplate(gray, board_template.img, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
    board_top_left = min_loc
    board_img = rescaled[ board_top_left[1] : board_top_left[1] + board_template.height, board_top_left[0]:board_top_left[0] + board_template.width]

    gray_board = cv.cvtColor(board_img, cv.COLOR_BGR2GRAY)
    hsv_board = cv.cvtColor(board_img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv.inRange(hsv_board, lower_blue, upper_blue) 
    gray_blue_board = gray_board.copy()
    gray_blue_board[mask==0] = 255

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv.inRange(hsv_board, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_board, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)
    gray_red_board = gray_board.copy()
    gray_red_board[mask==0] = 255

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv.inRange(hsv_board, lower_green, upper_green)
    gray_green_board = gray_board.copy()
    gray_green_board[mask==0] = 255

    pawn_gray_boards = [gray_blue_board, gray_red_board, gray_green_board]

    for i, (pts, pawn_board) in enumerate(zip(pawn_templates, pawn_gray_boards)):
        for pawn_template in pts:
            match = cv.matchTemplate(pawn_board, pawn_template.img, cv.TM_SQDIFF)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
            pawn_top_left = min_loc
            bottom_right = (pawn_top_left[0] + pawn_template.width, pawn_top_left[1] + pawn_template.height)
            if min_val < PAWN_THRESHOLD:
                cv.rectangle(board_img, pawn_top_left, bottom_right, 255, 1)
                field_idx = get_field((pawn_top_left[0] + pawn_template.width/2,pawn_top_left[1] + pawn_template.height/2))
                if field_idx != -1:
                    pawn_pos_votes[i][pawn_votes_idx[i]] = field_idx
                    pawn_votes_idx[i] = (pawn_votes_idx[i] + 1) % N_PAWN_VOTES
                    new_pos = statistics.mode(pawn_pos_votes[i])
                    if new_pos != pawn_pos[i]:
                        if pawn_pos[i] != -1:
                            roll = new_pos - pawn_pos[i]
                            if roll < 0:
                                roll = 40 + roll
                            secs = elapsed_frames // 30
                            mins = secs // 60
                            secs %= 60
                            time_str = f"[{mins:02d}:{secs:02d}]:"
                            events.append(f"{time_str} Player {player_names[i]} rolled: {roll}")
                            events.append(f"{time_str} Player {player_names[i]} moved from {field_names[pawn_pos[i]]} to {field_names[new_pos]}")
                        pawn_pos[i] = new_pos
                break

    pad_right = 800
    padded_image = cv.copyMakeBorder(rescaled, 0, 0, 0, pad_right, cv.BORDER_CONSTANT, None, (255, 255, 255))
    font = cv.FONT_HERSHEY_PLAIN
    fontScale = 1
    fontColor = (15,15,15)
    thickness = 1
    line_height = int( padded_image.shape[0] * 0.03)
    line_offset_y = line_height
    line_offset_x = int(rescaled.shape[1] + 0.05 * pad_right)
    lineType = 1

    n_players = 1
    for i in range(3):
        if pawn_pos[i] !=-1:
            bottomLeftCornerOfText = (line_offset_x, line_height * (n_players) + line_offset_y)
            text = f"Player {player_names[i]}: {field_names[pawn_pos[i]]}"
            n_players +=1
            cv.putText(padded_image, text,
                       bottomLeftCornerOfText,
                       font,
                       fontScale,
                       fontColor,
                       thickness,
                       lineType)
    if len(events) > 0:
        bottomLeftCornerOfText = (line_offset_x, line_height * (n_players) + line_offset_y)
        cv.putText(padded_image, "Events:",
                   bottomLeftCornerOfText,
                   font,
                   fontScale,
                   fontColor,
                   thickness,
                   lineType)
    for i in range(len(events)):
        if (i > 15):
            break
        bottomLeftCornerOfText = (line_offset_x, line_height * (n_players + i + 1) + line_offset_y)
        event = events[len(events)-i-1]
        cv.putText(padded_image, event,
                   bottomLeftCornerOfText,
                   font,
                   fontScale,
                   fontColor,
                   thickness,
                   lineType)


    cv.imshow('frame', padded_image)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



