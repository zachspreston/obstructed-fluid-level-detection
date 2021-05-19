import cv2
import numpy as np
import imutils
import math



def get_difference_img(bg_img, bottle_img, blur_kern=7):
    #smooth both bottle img and bg img
    
    # smooth image using 7x7 pixel gaussian kernel to remove a bit of noise
    #bg_img = cv2.GaussianBlur(bg_img, (blur_kern, blur_kern), 0)
    #bottle_img = cv2.GaussianBlur(bottle_img, (blur_kern, blur_kern), 0)

    # convert the images to grayscale
    blw_control_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    blw_bottle_img = cv2.cvtColor(bottle_img, cv2.COLOR_BGR2GRAY)


    diff_img = cv2.subtract(blw_control_img, blw_bottle_img)
    
    # smooth image using 7x7 pixel gaussian kernel to remove a bit of noise
    diff_img_smoothed = cv2.GaussianBlur(diff_img, (blur_kern, blur_kern), 0)

    return diff_img_smoothed

