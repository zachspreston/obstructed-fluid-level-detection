import cv2
import numpy as np
import imutils
import math
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

from contour_class import Bottle
from contour_extraction import *
from neck_removal import *
from liquid_level_detection import *
from calc_lv_percentage import *
from plot_results import *

# Old version
# PX_MM_CONVERSION = 0.0034
# For watershead resolution
PX_MM_CONVERSION = 0.0023


# Returns differenced monotone image of the bottle
def find_bottle_diff_img(control_img, bottle_img):
    # convert the images to grayscale
    blw_control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2GRAY)
    blw_bottle_img = cv2.cvtColor(bottle_img, cv2.COLOR_BGR2GRAY)

    # smooth image using 7x7 pixel gaussian kernel to remove a bit of noise
    #blw_bottle_img = cv2.GaussianBlur(blw_bottle_img, (9, 9), 0)

    diff_img = cv2.subtract(blw_control_img, blw_bottle_img)
    #diff_img = cv2.bitwise_not(diff_img)

    return diff_img















#TODO: make all functions pass the bottle object back

def main():
    control_img = cv2.imread("./bottle_images/no_bottle.png")
    bottle_img = cv2.imread("./bottle_images/bottle_blank.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_whiskey.png")
    #bottle_img = cv2.imread("./bottle_images/drink-bottle-green.png")
    bottle_img = cv2.imread("./bottle_images/drink-bottle.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_metal.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_green_off_centre.png")
    bottle_img = cv2.imread("./bottle_images/cola.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_green_half.png")
    bottle_img = cv2.imread("./bottle_images/water-glass-1.png")
    #bottle_img = cv2.imread("./bottle_images/wine_transparent.png")

    # #control_img = cv2.imread("./bottle_images/real_yellow_background.png")
    # #bottle_img = cv2.imread("./bottle_images/yellow_isolated.png")


    control_img = cv2.imread("./label_bottle_images/control_img.png")
    
    bottle_img = cv2.imread("./label_bottle_images/green_bottle_v1.png")
    bottle_img = cv2.imread("./label_bottle_images/green_bottle_v_p_covered.png")
    
    bottle_img = cv2.imread("./label_bottle_images/yellow_bottle_v1.png")
    bottle_img = cv2.imread("./label_bottle_images/yellow_bottle_v_f_covered.png")
    bottle_img = cv2.imread("./label_bottle_images/yellow_bottle_v_p_covered.png")
    #bottle_img = cv2.imread("./label_bottle_images/watershread_rect.png")

    bottle_img = cv2.imread("./label_bottle_images/conicol_flask_pink.png")
    bottle_img = cv2.imread("./label_bottle_images/conicol_flask_blue.png")

    bottle = Bottle(control_img, bottle_img)


    #thresh_bottle_img = find_ssid_thresh_bottle(control_img, bottle_img)
    ##____________Get Difference Image__________________##
    bottle.diff_img = find_bottle_diff_img(control_img, bottle_img)


    ##____________Get Threshold from Difference_________##
    #Use this for non-transparent bottles
    bottle.threshold_bottle_img = find_thresh_bottle(bottle.diff_img)


    ##____________Get Contour from Threshold_________##
    #bottle_contours = find_bottle_contour(bottle.threshold_bottle_img)
    bottle_contours = find_thresh_morph_canny(bottle, itera=15, c_thresh_1=16, c_thresh_2=60)
    bottle.raw_contour = bottle_contours[0]
    
    bottle = remove_neck_from_bottle_contour(bottle)



    ##____________Determine label + bottle contour_________##
    bottle = create_bottle_mask(bottle)


    ##____________Determine water level_________##
    #bottle = determine_fluid_level(bottle)
    bottle = watershread_fluid_level_detect(bottle)


    ##_____________Calculate volumes____________##
    bottle.volume_ml = px_area_to_standard_volume(bottle.processed_contour, PX_MM_CONVERSION, bottle_shape='c', standardise_ml = True)
    
    if (bottle.has_fluid_level):
        print('calculating fluid volume')
        bottle.fluid_ml = px_area_to_standard_volume(bottle.fluid_level_processed_contour, PX_MM_CONVERSION, bottle_shape='c', standardise_ml = False)
    print(bottle.volume_ml)
    
    


    ##_____________Render results____________##
    bottle = draw_bottle_and_fluid_contours_and_labels(bottle)
    plot_cv_results_label_removal(bottle)
    
    
    #cv2.imshow("bottle", bottle_img)
    #cv2.waitKey(0)


main()