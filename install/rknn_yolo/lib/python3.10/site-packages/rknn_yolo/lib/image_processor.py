import cv2
import numpy as np

def process_image(image, is_turning=False, turn_direction=0):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([155, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    line_center = None
    center_y = None
    bounding_box = None
    current_mask = None

    if not is_turning:
        roi_line = image[int(h*0.75):h, :, :]
        h_roi_line, w_roi_line = roi_line.shape[:2]
        hsv_line = cv2.cvtColor(roi_line, cv2.COLOR_BGR2HSV)
        mask1_line = cv2.inRange(hsv_line, lower_red1, upper_red1)
        mask2_line = cv2.inRange(hsv_line, lower_red2, upper_red2)
        mask_line = mask1_line | mask2_line
        mask_line = cv2.morphologyEx(mask_line, cv2.MORPH_OPEN, kernel)
        contours_line, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_mask = mask_line

        if contours_line:
            max_contour = max(contours_line, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 100:
                fixed_y_roi = h_roi_line // 2
                center_y = int(h * 0.875)
                horizontal_line = mask_line[fixed_y_roi, :]
                white_pixels = np.where(horizontal_line > 0)[0]
                if len(white_pixels) > 0:
                    line_center = int(np.mean(white_pixels))
                    x, y, w, h = cv2.boundingRect(max_contour)
                    if w > 20:
                        y += int(h*0.75)
                        bounding_box = (x, y, w, h)
    else:
        roi_turn = image[0:int(h*0.5), :, :]
        h_roi_turn, w_roi_turn = roi_turn.shape[:2]
        hsv_turn = cv2.cvtColor(roi_turn, cv2.COLOR_BGR2HSV)
        mask1_turn = cv2.inRange(hsv_turn, lower_red1, upper_red1)
        mask2_turn = cv2.inRange(hsv_turn, lower_red2, upper_red2)
        mask_turn = mask1_turn | mask2_turn
        mask_turn = cv2.morphologyEx(mask_turn, cv2.MORPH_OPEN, kernel)
        contours_turn, _ = cv2.findContours(mask_turn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_mask = mask_turn

        if contours_turn:
            max_contour = max(contours_turn, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 100:
                fixed_y_roi = h_roi_turn // 2
                center_y = int(h * 0.25)
                if turn_direction == -1:
                    horizontal_line = mask_turn[fixed_y_roi, 0:w_roi_turn//2]
                    white_pixels = np.where(horizontal_line > 0)[0]
                elif turn_direction == 1:
                    horizontal_line = mask_turn[fixed_y_roi, w_roi_turn//2:w_roi_turn]
                    white_pixels = np.where(horizontal_line > 0)[0] + w_roi_turn//2
                else:
                    horizontal_line = mask_turn[fixed_y_roi, :]
                    white_pixels = np.where(horizontal_line > 0)[0]

                if len(white_pixels) > 0:
                    line_center = int(np.mean(white_pixels))
                    x, y, w, h = cv2.boundingRect(max_contour)
                    bounding_box = (x, y, w, h)

    return line_center, center_y, current_mask, bounding_box