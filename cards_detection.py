import cv2
import numpy as np

CHANCE_CARD_DIM_RATIO = 1.5
MIN_AREA = 15000
MAX_AREA = 30000
MINIMAL_DETECTION_RATIO = 0.4
OTHER_COLOR_NUMBER_THRESHOLD = 1000
MARGIN_RATIO = 0.1

frame_count = 0

def filter_hsv_values(hsv_roi):
    """ filters HSV values and returns masks for red and blue cards """
    hsv_ranges = {
        "red": [(np.array([0, 40, 90]), np.array([10, 90, 255])),
                (np.array([0, 70, 92]), np.array([18, 100, 255]))],
        "blue": [(np.array([40, 0, 90]), np.array([120, 120, 200]))],
        "hand": [(np.array([10, 26, 0]), np.array([34, 157, 244]))],
        "background": [(np.array([0, 8, 35]), np.array([18, 22, 255]))]
    }

    def combine_masks(ranges):
        combined_mask = cv2.inRange(hsv_roi, ranges[0][0], ranges[0][1])
        for lower, upper in ranges[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv_roi, lower, upper))
        return combined_mask
    
    mask_red = combine_masks(hsv_ranges["red"])
    mask_blue = combine_masks(hsv_ranges["blue"])
    hand_mask = combine_masks(hsv_ranges["hand"])
    mask_background = combine_masks(hsv_ranges["background"])

    mask_red &= ~hand_mask
    mask_red &= ~mask_background
    mask_blue &= ~mask_background

    return mask_red, mask_blue


def detect_squares(frame):
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
    red_flag, blue_flag = False, False # there cannot be multiple such detections
    gray = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 20, 120)

    # closing
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    margin_w, margin_h = int(width * MARGIN_RATIO), int(height * MARGIN_RATIO)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if x < margin_w or x + w > width - margin_w or y < margin_h or y + h > height - margin_h:
            continue
        
        if w * h < MIN_AREA or w * h > MAX_AREA:
            continue
        
        detections = np.sum(edges[y:y+h, x:x+w] == 255)
        
        if detections / (w * h) < MINIMAL_DETECTION_RATIO:
            continue
        
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_red, mask_blue = filter_hsv_values(hsv_roi)
        red_pixels = np.sum(mask_red == 255)
        blue_pixels = np.sum(mask_blue == 255)

        if red_pixels > (MIN_AREA / 5) and blue_pixels < OTHER_COLOR_NUMBER_THRESHOLD and w/h < CHANCE_CARD_DIM_RATIO and not red_flag:
            red_flag = True
            cv2.putText(frame, f"Red card {w * h} {red_pixels}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        elif blue_pixels > (MIN_AREA / 5) and red_pixels < OTHER_COLOR_NUMBER_THRESHOLD and w/h < CHANCE_CARD_DIM_RATIO and not blue_flag:
            blue_flag = True
            cv2.putText(frame, f"Blue card {w * h} {blue_pixels}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

cap = cv2.VideoCapture("output4.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = detect_squares(frame)
    cv2.imshow("Squares Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
