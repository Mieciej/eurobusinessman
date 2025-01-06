import numpy as np
import cv2 as cv
import glob
import statistics
from dataclasses import dataclass
from cards_detection import detect_chance_cards

## PARAMETERS
N_PAWN_VOTES = 101
PAWN_THRESHOLD = 2000000

@dataclass
class Template:
    img: np.ndarray
    width: int
    height: int


blue_pawn_template = Template(cv.imread("blue_pawn.png", cv.IMREAD_GRAYSCALE), 32, 32)
red_pawn_template = Template(cv.imread("red_pawn.png", cv.IMREAD_GRAYSCALE), 32, 32)
green_pawn_template = Template(cv.imread("green_pawn.png", cv.IMREAD_GRAYSCALE), 32, 32)
blue_pawn_on_blue_template = Template(cv.imread("blue_pawn_on_blue.png", cv.IMREAD_GRAYSCALE), 18, 18)

blue_pawn_hard_template = Template(cv.imread("blue_pawn_hard.png", cv.IMREAD_GRAYSCALE), 32, 32)
green_pawn_hard_template = Template(cv.imread("green_pawn_hard.png", cv.IMREAD_GRAYSCALE), 32, 32)
blue_pawn_hard_close_template = Template(cv.imread("blue_pawn_hard_close.png", cv.IMREAD_GRAYSCALE), 32, 32)
green_pawn_hard_close_template = Template(cv.imread("green_pawn_hard_close.png", cv.IMREAD_GRAYSCALE), 32, 32)

pawn_templates = None
board_easy_template = Template(cv.imread("board.png", cv.IMREAD_GRAYSCALE), 620, 620)
board_medium_template = Template(cv.imread("board_medium.png", cv.IMREAD_GRAYSCALE), 630, 610)
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

red_chance_fields = [7, 22, 36]
blue_chance_fields = [2, 17, 33]
tax_field = 38
paid_parking = 4
free_parking = 20
go_to_prison = 30
prison = 10
special_fields = red_chance_fields + blue_chance_fields + [tax_field, paid_parking, free_parking, go_to_prison, prison, 0]
field_owner = [-1] * 40
fields_owned = {'blue': [], 'red': [], 'green': []}

def get_field_coords_easy(field_idx):
    field_coords = [
        ((17, 532), (97, 611)), ((16, 486), (78, 526)), ((15, 439), (97, 480)), ((14, 392), (77, 433)), ((13, 345), (95, 387)),
        ((11, 298), (94, 340)), ((10, 250), (75, 292)), ((10, 203), (93, 245)), ((7, 154), (72, 196)), ((6, 105), (71, 148)),
        ((4, 14), (90, 100)), ((93, 14), (137, 79)), ((142, 13), (186, 99)), ((190, 13), (234, 79)), ((238, 13), (282, 79)),
        ((288, 13), (330, 98)), ((336, 12), (379, 78)), ((385, 13), (427, 98)), ((434, 12), (476, 78)), ((482, 12), (524, 78)),
        ((531, 11), (613, 97)), ((549, 103), (612, 146)), ((549, 153), (611, 195)), ((548, 201), (610, 243)), ((545, 249), (608, 291)),
        ((526, 298), (608, 339)), ((545, 345), (607, 386)), ((544, 392), (606, 433)), ((524, 439), (605, 480)), ((543, 487), (605, 527)),
        ((523, 532), (603, 612)), ((476, 550), (517, 611)), ((429, 550), (470, 612)), ((382, 531), (423, 611)), ((336, 550), (377, 611)),
        ((289, 531), (330, 612)), ((242, 550), (284, 611)), ((195, 550), (237, 610)), ((149, 531), (191, 611)), ((102, 550), (144, 611))
    ]
    return field_coords[field_idx]

def get_field_coords_medium(field_idx):
    field_coords = [
        ((17, 506), (102, 591)), ((19, 458), (104, 506)), ((20, 411), (104, 458)), ((22, 364), (105, 411)), ((23, 317), (106, 364)),
        ((25, 270), (108, 317)), ((26, 225), (109, 271)), ((27, 181), (110, 225)), ((29, 133), (112, 179)), ((29, 87), (112, 133)),
        ((30, 4), (113, 86)), ((113, 4), (160, 86)), ((160, 4), (205, 86)), ((205, 5), (250, 87)), ((250, 5), (296, 87)),
        ((296, 5), (342, 87)), ((342, 6), (388, 88)), ((388, 6), (434, 88)), ((434, 6), (480, 88)), ((480, 7), (526, 89)),
        ((526, 8), (611, 90)), ((526, 90), (611, 137)), ((526, 137), (611, 182)), ((526, 182), (611, 229)), ((526, 229), (611, 277)),
        ((526, 277), (611, 322)), ((526, 322), (611, 371)), ((526, 371), (611, 416)), ((526, 416), (611, 465)), ((526, 465), (611, 514)),
        ((526, 514), (611, 600)), ((481, 514), (527, 600)), ((432, 513), (481, 599)), ((385, 513), (432, 598)), ((338, 512), (385, 598)),
        ((291, 511), (338, 597)), ((244, 510), (291, 596)), ((196, 509), (244, 595)), ((150, 509), (196, 594)), ((102, 508), (150, 593))
    ]
    coords = field_coords[field_idx]
    coords = ((coords[0][0], coords[0][1] + 15), (coords[1][0], coords[1][1] + 15))
    return coords

def get_field_coords_hard(field_idx):
    field_coords = [
        ((508, 118), (557, 185)), ((520, 197), (557, 240)), ((509, 239), (558, 281)), ((522, 284), (559, 322)), ((510, 327), (560, 364)),
        ((510, 370), (560, 406)), ((524, 413), (560, 448)), ((512, 458), (560, 490)), ((525, 500), (562, 531)), ((525, 544), (563, 574)),
        ((512, 591), (564, 653)), ((480, 618), (510, 672)), ((448, 608), (478, 683)), ((411, 637), (444, 693)), ((372, 648), (409, 705)),
        ((331, 638), (368, 719)), ((286, 671), (326, 732)), ((239, 660), (281, 748)), ((185, 712), (232, 764)), ((127, 712), (179, 781)),
        ((9, 716), (121, 799)), ((9, 648), (98, 689)), ((10, 580), (123, 621)), ((12, 580), (98, 561)), ((13, 447), (101, 498)),
        ((14, 384), (126, 435)), ((15, 316), (103, 374)), ((17, 251), (105, 313)), ((19, 188), (128, 253)), ((22, 123), (105, 191)),
        ((22, 8), (129, 134)), ((136, 34), (184, 119)), ((191, 46), (235, 128)), ((241, 57), (282, 158)), ((288, 68), (326, 144)),
        ((331, 77), (367, 170)), ((372, 88), (405, 176)), ((409, 95), (441, 163)), ((444, 103), (474, 186)), ((476, 110), (506, 174))
    ]
    return field_coords[field_idx]

def get_field_easy(pos):
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

def get_field_medium(pos):
    x, y = pos
    if 17 <= x and 506 <= y and x <= 102  and y <= 591:
        return 0
    elif 19 <= x and 458 <= y and x <= 104  and y <= 506:
        return 1
    elif 20 <= x and 411 <= y and x <= 104  and y <= 458:
        return 2
    elif 22 <= x and 364 <= y and x <= 105  and y <= 411:
        return 3
    elif 23 <= x and 317 <= y and x <= 106  and y <= 364:
        return 4
    elif 25 <= x and 270 <= y and x <= 108  and y <= 317:
        return 5
    elif 26 <= x and 225 <= y and x <= 109  and y <= 271:
        return 6
    elif 27 <= x and 181 <= y and x <= 110  and y <= 225:
        return 7
    elif 29 <= x and 133 <= y and x <= 112  and y <= 179:
        return 8
    elif 29 <= x and 87 <= y and x <= 112  and y <= 133:
        return 9
    elif 30 <= x and 4 <= y and x <= 113  and y <= 86:
        return 10
    elif 113 <= x and 4 <= y and x <= 160  and y <= 86:
        return 11
    elif 160 <= x and 4 <= y and x <= 205  and y <= 86:
        return 12
    elif 205 <= x and 5 <= y and x <= 250  and y <= 87:
        return 13
    elif 250 <= x and 5 <= y and x <= 296  and y <= 87:
        return 14
    elif 296 <= x and 5 <= y and x <= 342  and y <= 87:
        return 15
    elif 342 <= x and 6 <= y and x <= 388  and y <= 88:
        return 16
    elif 388 <= x and 6 <= y and x <= 434  and y <= 88:
        return 17
    elif 434 <= x and 6 <= y and x <= 480  and y <= 88:
        return 18
    elif 480 <= x and 7 <= y and x <= 526  and y <= 89:
        return 19
    elif 526 <= x and 8 <= y and x <= 611  and y <= 90:
        return 20
    elif 526 <= x and 90 <= y and x <= 611  and y <= 137:
        return 21
    elif 526 <= x and 137 <= y and x <= 611  and y <= 182:
        return 22
    elif 526 <= x and 182 <= y and x <= 611  and y <= 229:
        return 23
    elif 526 <= x and 229 <= y and x <= 611  and y <= 277:
        return 24
    elif 526 <= x and 277 <= y and x <= 611  and y <= 322:
        return 25
    elif 526 <= x and 322 <= y and x <= 611  and y <= 371:
        return 26
    elif 526 <= x and 371 <= y and x <= 611  and y <= 416:
        return 27
    elif 526 <= x and 416 <= y and x <= 611  and y <= 465:
        return 28
    elif 526 <= x and 465 <= y and x <= 611  and y <= 514:
        return 29
    elif 526 <= x and 514 <= y and x <= 611  and y <= 600:
        return 30
    elif 481 <= x and 514 <= y and x <= 527  and y <= 600:
        return 31
    elif 432 <= x and 513 <= y and x <= 481  and y <= 599:
        return 32
    elif 385 <= x and 513 <= y and x <= 432  and y <= 598:
        return 33
    elif 338 <= x and 512 <= y and x <= 385  and y <= 598:
        return 34
    elif 291 <= x and 511 <= y and x <= 338  and y <= 597:
        return 35
    elif 244 <= x and 510 <= y and x <= 291  and y <= 596:
        return 36
    elif 196 <= x and 509 <= y and x <= 244  and y <= 595:
        return 37
    elif 150 <= x and 509 <= y and x <= 196  and y <= 594:
        return 38
    elif 102 <= x and 508 <= y and x <= 150  and y <= 593:
        return 39
    else:
        return -1

def get_field_hard(pos):
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
get_field_coords = None

cap = cv.VideoCapture("output1.mp4")

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * 0.3) + 800
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * 0.3)
fps = cap.get(cv.CAP_PROP_FPS)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output_with_detection.mp4', fourcc, fps, (width, height))


events = []
pawn_playing = [False, False, False]
pawn_started_on_start_field = [False, False, False]

player_names = [ "blue",
                 "red",
                 "green"]

elapsed_frames = 0
board_template = None
game_start = False


ret, frame = cap.read()
rescaled = cv.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv.INTER_CUBIC)

# save the frame
cv.imwrite('frame.png', rescaled)

# rescaled = cv.flip(rescaled, -1)
gray = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
match = cv.matchTemplate(gray, board_easy_template.img, cv.TM_SQDIFF)
min_val_easy, _, _, _ = cv.minMaxLoc(match)
match = cv.matchTemplate(gray, board_medium_template.img, cv.TM_SQDIFF)
min_val_medium, _, _, _ = cv.minMaxLoc(match)
match = cv.matchTemplate(gray, board_hard_template.img, cv.TM_SQDIFF)
min_val_hard, _, _, _ = cv.minMaxLoc(match)
if min_val_easy <= min_val_medium and min_val_easy <= min_val_hard:
    board_template = board_easy_template
    pawn_templates = [[blue_pawn_template], [red_pawn_template], [green_pawn_template]]
    get_field = get_field_easy
    get_field_coords = get_field_coords_easy
    print("The board is easy")
elif min_val_medium < min_val_easy and min_val_medium < min_val_hard:
    board_template = board_medium_template
    pawn_templates = [[blue_pawn_template, blue_pawn_on_blue_template], [red_pawn_template], [green_pawn_template]]
    get_field = get_field_medium
    get_field_coords = get_field_coords_medium
    print("The board is medium")
else:
    board_template = board_hard_template
    pawn_templates = [[blue_pawn_hard_template, blue_pawn_hard_close_template], [red_pawn_template], [green_pawn_hard_template, green_pawn_hard_close_template] ]
    get_field = get_field_hard
    get_field_coords = get_field_coords_hard
    print("The board is hard!")

prev_board_top_left = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    elapsed_frames +=1
    secs = elapsed_frames // 30
    mins = secs // 60
    secs %= 60
    time_str = f"[{mins:02d}:{secs:02d}]:"
    if not game_start:
        n_playing = sum(pawn_playing)
        if n_playing > 1:
            game_start = n_playing == sum(pawn_started_on_start_field)
            if game_start:
                events.append(f"{time_str} game started")

    rescaled = cv.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv.INTER_CUBIC)
    if get_field == get_field_medium:
        rescaled = cv.flip(rescaled, -1)

    gray = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    
    rescaled = detect_chance_cards(rescaled)

    match = cv.matchTemplate(gray, board_template.img, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
    current_board_top_left = min_loc
    
    if prev_board_top_left is not None:
        dx = current_board_top_left[0] - prev_board_top_left[0]
        dy = current_board_top_left[1] - prev_board_top_left[1]
        if abs(dx) > 100 or abs(dy) > 100:
            current_board_top_left = prev_board_top_left
        else:
            prev_board_top_left = current_board_top_left
    else:
        prev_board_top_left = current_board_top_left
    
    board_top_left = current_board_top_left
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
                    pawn_playing[i] = True
                    pawn_pos_votes[i][pawn_votes_idx[i]] = field_idx
                    pawn_votes_idx[i] = (pawn_votes_idx[i] + 1) % N_PAWN_VOTES
                    new_pos = statistics.mode(pawn_pos_votes[i])
                    if new_pos != pawn_pos[i]:
                        if pawn_pos[i] != -1:
                            roll = new_pos - pawn_pos[i]
                            went_throug_start = False
                            if roll < 0:
                                went_throug_start = True
                                roll = 40 + roll
                            if 2 <= roll and roll <= 12:
                                events.append(f"{time_str} Player {player_names[i]} rolled: {roll}")
                            events.append(f"{time_str} Player {player_names[i]} moved from {field_names[pawn_pos[i]]} to {field_names[new_pos]}")
                            if went_throug_start:
                                events.append(f"{time_str} Player {player_names[i]} went through {field_names[0]} and received $400")
                            if new_pos in blue_chance_fields:
                                events.append(f"{time_str} Player {player_names[i]} has to draw a blue chance card")
                            if new_pos in red_chance_fields:
                                events.append(f"{time_str} Player {player_names[i]} has to draw a red chance card")
                            if new_pos == tax_field:
                                events.append(f"{time_str} Player {player_names[i]} paid $200 tax")

                            if new_pos == paid_parking:
                                events.append(f"{time_str} Player {player_names[i]} paid $400 parking fee")

                            if game_start and new_pos not in special_fields:
                                owner = field_owner[new_pos]
                                if  owner == -1:
                                    events.append(f"{time_str} Player {player_names[i]} bought {field_names[new_pos]}")
                                    field_owner[new_pos] = i
                                    fields_owned[player_names[i]].append(new_pos)
                                elif owner != i:
                                    events.append(f"{time_str} Player {player_names[i]} has to pay rent to player {player_names[owner]}")
                                elif owner == i:
                                    events.append(f"{time_str} Player {player_names[i]} is visting their property")
                        else:
                            pawn_started_on_start_field[i] = new_pos == 0
                        pawn_pos[i] = new_pos
                break

    pad_right = 800
    padded_image = cv.copyMakeBorder(rescaled, 0, 0, 0, pad_right, cv.BORDER_CONSTANT, None, (255, 255, 255))
    font = cv.FONT_HERSHEY_PLAIN
    fontScale = 1
    fontColor = (15,15,15)
    fontPlayerColor = [(254, 50, 50), (33, 30, 200), (88, 169, 16)]
    thickness = 1
    thicknessImportant = 2
    line_height = int( padded_image.shape[0] * 0.03)
    line_offset_y = line_height
    line_offset_x = int(rescaled.shape[1] + 0.05 * pad_right)
    lineType = 1

    n_players = 1
    for i in range(3):
        if pawn_pos[i] != -1:
            column = 300 if n_players % 2 == 0 else 0
            bottomLeftCornerOfText = (line_offset_x + column, line_height + line_offset_y)
            
            player_text = f"Player {player_names[i].upper()}"
            rest_text = f": {field_names[pawn_pos[i]]}"
            
            player_text_size = cv.getTextSize(player_text, font, fontScale, thickness)[0]
            player_text_width = player_text_size[0]

            cv.putText(padded_image, player_text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontPlayerColor[i],
                    thicknessImportant,
                    lineType)
            
            rest_text_position = (bottomLeftCornerOfText[0] + player_text_width, bottomLeftCornerOfText[1])
            cv.putText(padded_image, rest_text,
                    rest_text_position,
                    font,
                    fontScale,
                    (0, 0, 0),
                    thickness,
                    lineType)
            
            n_players += 1
            
    if len(events) > 0:
        bottomLeftCornerOfText = (line_offset_x, line_height * (n_players) + line_offset_y)
        cv.putText(padded_image, "Events:",
                   bottomLeftCornerOfText,
                   font,
                   fontScale * 1.4,
                   fontColor,
                   thickness * 2,
                   lineType)
    for i in range(len(events)):
        if (i > 12):
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

    n_players = 1
    for i in range(3):
        if pawn_pos[i] == -1:
            continue
        alpha = 0.4
        player = player_names[i]
        player_str = player.split()[0].upper()
        column = 300 if n_players % 2 == 0 else 0
        bottomLeftCornerOfText = (line_offset_x + column, line_height + 600)
        cv.putText(padded_image, f"{player_str} properties:",
                   bottomLeftCornerOfText,
                   font,
                   fontScale,
                   fontPlayerColor[i],
                   thicknessImportant,
                   lineType)
        for j, field in enumerate(fields_owned[player]):
            field_upper_left, field_bottom_right = get_field_coords(field)
            field_upper_left = (field_upper_left[0] + board_top_left[0], field_upper_left[1] + board_top_left[1])
            field_bottom_right = (field_bottom_right[0] + board_top_left[0], field_bottom_right[1] + board_top_left[1])

            overlay = padded_image.copy()
            cv.rectangle(overlay, field_upper_left, field_bottom_right, fontPlayerColor[i], -1)
            cv.addWeighted(overlay, alpha, padded_image, 1 - alpha, 0, padded_image)
            
            column = 300 if n_players % 2 == 0 else 0
            bottomLeftCornerOfText = (line_offset_x + column, line_height * (j + 2) + 600)
            cv.putText(padded_image, field_names[field],
                       bottomLeftCornerOfText,
                       font,
                       fontScale,
                       fontColor,
                       thickness,
                       lineType)
        n_players += 1
    
    out.write(padded_image)
    
    # cv.imshow('frame', padded_image)
    if cv.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv.destroyAllWindows()



