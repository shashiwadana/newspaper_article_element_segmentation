import os

import cv2
import math
import numpy as np

def preprocessing(img,num):
    print("pre processing...")
    image= img
    # gray convertion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # threshold
    thresh_x = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_OTSU)[1]
    med_x = cv2.medianBlur(thresh_x, 3)
    thresh_y = cv2.threshold(abs_grad_y, 0, 255, cv2.THRESH_OTSU)[1]
    med_y = cv2.medianBlur(thresh_y, 3)
    #cv2.imwrite('x.jpg',abs_grad_x)
    #cv2.imwrite('y.jpg', abs_grad_y)

    # bluring
    kernel_size = 3
    blur_thresh_x = cv2.GaussianBlur(med_x, (kernel_size, kernel_size), 0)
    blur_thresh_y = cv2.GaussianBlur(med_y, (kernel_size, kernel_size), 0)
    #cv2.imwrite('xg.jpg', blur_thresh_x)
    #cv2.imwrite('yg.jpg', blur_thresh_y)
    # Run Hough on edge detected image
    width = image.shape[1]
    height = image.shape[0]
    rho = 1  # distance resolution in pixels of the Hough grid
    theta_vertical = np.pi  # angular resolution in radians of the Hough grid
    theta_horizonatal = np.pi / 2
    threshold_horizontal = math.floor(width * 0.2)  # minimum number of votes (intersections in Hough grid cell)
    threshold_vertical = math.floor(height * 0.2)
    min_line_length_horizontal = math.floor(width * 0.2)
    min_line_length_vertical = math.floor(height * 0.2)  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray) * 0  # creating a blank to draw lines on

    # Vertical lines
    vertical_lines = cv2.HoughLinesP(blur_thresh_x, rho, theta_vertical, threshold_vertical, np.array([]),
                                     min_line_length_vertical, max_line_gap)

    if vertical_lines is not None:
        for line in vertical_lines:
            for x1, y1, x2, y2 in line:
                # here it's possible to add a selection of only vertical lines
                if np.abs(y1 - y2) > 0.1 * np.abs(x1 - x2):
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, 5)

    # Horizontal lines
    horizontal_lines = cv2.HoughLinesP(blur_thresh_y, rho, theta_horizonatal, threshold_horizontal, np.array([]),
                                       min_line_length_horizontal, max_line_gap)

    if horizontal_lines is not None:
        for line in horizontal_lines:
            for x1, y1, x2, y2 in line:
                # here it's possible to add a selection of only horizontal lines
                if np.abs(x1 - x2) > 0.1 * np.abs(y1 - y2):
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, 5)

    # threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61,18)


    # remove lines
    clean_thresh = cv2.subtract(thresh, line_image)
    clean_invert = cv2.bitwise_not(clean_thresh)
    #cv2.imwrite('l.jpg',line_image)
    cv2.namedWindow('detectedLines', cv2.WINDOW_NORMAL)
    cv2.imshow('detectedLines', line_image)
    cv2.namedWindow('preProcessed', cv2.WINDOW_NORMAL)
    cv2.imshow('preProcessed', clean_invert)
    cv2.waitKey(0)
    name = "C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/" + num
    cv2.imwrite(os.path.join(name,'pp.jpg'), clean_invert)
    return  clean_invert