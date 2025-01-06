import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Filter HSV values")
parser.add_argument("image", type=str, help="Path to the image")
args = parser.parse_args()

image_path = args.image
img = cv2.imread(image_path)
if img is None:
    print("Image not found")
    exit()

trackbar_values = {
    "LowH": 0,
    "HighH": 0,
    "LowS": 0,
    "HighS": 0,
    "LowV": 0,
    "HighV": 0
}

def update_trackbar(value):
    for key in trackbar_values.keys():
        trackbar_values[key] = cv2.getTrackbarPos(key, "Control")

iLowH = 155
iHighH = 225
iLowS = 47
iHighS = 126
iLowV = 82
iHighV = 132

cv2.namedWindow("Control")
cv2.createTrackbar("LowH", "Control", iLowH, 255, update_trackbar)
cv2.createTrackbar("HighH", "Control", iHighH, 255, update_trackbar)
cv2.createTrackbar("LowS", "Control", iLowS, 255, update_trackbar)
cv2.createTrackbar("HighS", "Control", iHighS, 255, update_trackbar)
cv2.createTrackbar("LowV", "Control", iLowV, 255, update_trackbar)
cv2.createTrackbar("HighV", "Control", iHighV, 255, update_trackbar)

while True:
    lower = np.array([
        trackbar_values["LowH"],
        trackbar_values["LowS"],
        trackbar_values["LowV"]
    ], dtype="uint8")
    higher = np.array([
        trackbar_values["HighH"],
        trackbar_values["HighS"],
        trackbar_values["HighV"]
    ], dtype="uint8")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    flt = cv2.inRange(hsv, lower, higher)

    cv2.imshow("Filtered", flt)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
