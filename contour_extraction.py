import cv2
import numpy as np
import imutils
import math

# Method invoking Canny built in Term1 W6
def find_thresh_morph_canny(bottle, kern=4, itera=15, c_thresh_1 = 18, c_thresh_2 = 60):

    # Find edges
    cannyThreshold1 = c_thresh_1
    cannyThreshold2 = c_thresh_2
    #bottle.diff_img = cv2.bitwise_not(bottle.diff_img)
    bottle_edges_canny = cv2.Canny(bottle.diff_img, cannyThreshold1, cannyThreshold2)

    # Apply morphology
    kernel = np.ones((kern,kern),np.uint8)
    morphed_bottle_edges = cv2.morphologyEx(bottle_edges_canny, cv2.MORPH_CLOSE, kernel, iterations=itera)
    

    # Attempt to find contours
    contours = cv2.findContours(morphed_bottle_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # Draw on contours
    img = bottle.diff_img.copy()
    cv2.drawContours(img, contours, -1, (0,255,0), 2)
    # incase more than one difference, grab the largest one
    img_diff_areas = [cv2.contourArea(contour) for contour in contours]
    (contours_ordered, img_diff_areas) = zip(*sorted(zip(contours, img_diff_areas), key=lambda a:a[1]))
    
    sorted_contours = sorted(contours_ordered, key=cv2.contourArea, reverse= True)
    
    # Account for drift of the contours due to morphing iterations
    # If the image has even x or y dimensions, a centre pixel cannot be determined
    # so a drift will occur
    #TODO: this may be very computationally demanding. It may be worth only processing the largest contour
    # if (bottle.img_width % 2 == 0) or (bottle.img_height % 2 == 0):
    #     for contour in sorted_contours:
    #         for points in contour:
    #             if (bottle.img_width % 2 != 0):
    #                 points[0][0] -= itera
    #             if (bottle.img_height % 2 == 0):
    #                 points[0][1] -= itera

    for contour in sorted_contours:
        for points in contour:
            points[0][0] -= itera
            points[0][1] -= itera

    #Update bottle object
    bottle.threshold_bottle_img = cv2.bitwise_not(morphed_bottle_edges)
    bottle.canny_bottle_img = cv2.bitwise_not(bottle_edges_canny)
    
    return sorted_contours
    





##_________________________________Legacy (unused) methods______________________________________##

# Returns thresholded image of bottle
def find_thresh_bottle(diff_img):

    thresh_raw = cv2.threshold(diff_img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Print to visualise threshold
    #bottle_img[thresh_raw == 255] = [0, 0, 255]
    #cv2.imshow('threshold', thresh_raw)

    return thresh_raw



# Returns contour array for the bottle
def find_bottle_contour(thresh_img):
    
    # params: source image, retrieval mode, contour approximation method
    img_diff_contours = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_diff_contours = imutils.grab_contours(img_diff_contours)
    
    # incase more than one difference, grab the largest one
    img_diff_areas = [cv2.contourArea(contour) for contour in img_diff_contours]
    (img_diff_contours, img_diff_areas) = zip(*sorted(zip(img_diff_contours, img_diff_areas), key=lambda a:a[1]))

    return img_diff_contours