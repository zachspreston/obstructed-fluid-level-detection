import cv2
import numpy as np
import imutils
import math
import matplotlib.pyplot as plt

from contour_class import Bottle, extract_subcontour
from process_raw_diff import get_difference_img
from contour_extraction import *
from neck_removal import *
from liquid_level_detection import *
from calc_lv_percentage import *
from plot_results import *
from levelshred import levelshred

# Old version
#PX_MM_CONVERSION = 0.0034
# For watershead resolution
PX_MM_CONVERSION = 0.0021

def gen_square_contour(xy1=[], xy2=[]):
    return [np.array([[xy1[0], xy1[1]], [xy2[0],xy1[1]], [xy2[0],xy2[1]], [xy1[0], xy2[1]]], dtype=np.int32)]






def main():
    # Manual select of which image to process

    #control_img = cv2.imread("./bottle_images/no_bottle.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_blank.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_whiskey.png")
    #bottle_img = cv2.imread("./bottle_images/drink-bottle-green.png")
    #bottle_img = cv2.imread("./bottle_images/drink-bottle.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_metal.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_green_off_centre.png")
    #bottle_img = cv2.imread("./bottle_images/cola.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_green_half.png")
    #bottle_img = cv2.imread("./bottle_images/water-glass-1.png")
    #bottle_img = cv2.imread("./bottle_images/wine_transparent.png")

    #control_img = cv2.imread("./bottle_images/real_yellow_background.png")
    #bottle_img = cv2.imread("./bottle_images/yellow_isolated.png")


    #control_img = cv2.imread("./label_bottle_images/control_img.png")
    
    #bottle_img = cv2.imread("./label_bottle_images/green_bottle_v1.png")
    #bottle_img = cv2.imread("./label_bottle_images/green_bottle_v_p_covered.png")
    
    #ottle_img = cv2.imread("./label_bottle_images/yellow_bottle_v1.png")
    #bottle_img = cv2.imread("./label_bottle_images/yellow_bottle_v_f_covered.png")
    #bottle_img = cv2.imread("./label_bottle_images/yellow_bottle_v_p_covered.png")
    #bottle_img = cv2.imread("./label_bottle_images/watershread_rect.png")

    #bottle_img = cv2.imread("./label_bottle_images/conicol_flask_pink.png")
    #bottle_img = cv2.imread("./label_bottle_images/conicol_flask_blue.png")

    # control_img = cv2.imread("./live_bottle_images/black_nalgene_control.png")
    # bottle_img = cv2.imread("./live_bottle_images/black_nalgene_no_neck.png")

    # control_img = cv2.imread("./live_bottle_images/beaker_img_control.png")
    # bottle_img = cv2.imread("./live_bottle_images/beaker_img.png")
    
    # bottle_img = cv2.imread("./live_bottle_images/glare1.jpg")
    # h,w = bottle_img.shape[:2]
    # control_img = np.zeros((h,w,3), np.uint8)

    control_img = cv2.imread("./live_bottle_images/control_img.png")
    bottle_img = cv2.imread("./live_bottle_images/bottle_img.png")


    bottle = Bottle(control_img, bottle_img)
    
    # Manual label determination

    #____________Main label contours________________#
    # Yellow bottle with label
    #bottle.label_contour = [np.array([[357, 477], [493,477], [493,616], [357, 616]], dtype=np.int32)]

    # black nalgene label contour
    #bottle.label_contour = gen_square_contour(xy1=[590,270], xy2=[640,553])

    # blue nalgene label contour
    #bottle.label_contour = gen_square_contour(xy1=[499,451], xy2=[635,528])
    
    # soy sauce label contour
    #bottle.label_contour = gen_square_contour(xy1=[471,470], xy2=[540,778])
    
    # Milk label contour
    #bottle.label_contour = gen_square_contour(xy1=[531,555], xy2=[758,774])
    
    # Scorpion label contour
    #bottle.label_contour = gen_square_contour(xy1=[576,423], xy2=[649,757])

    # Salt label contour
    #bottle.label_contour = gen_square_contour(xy1=[545, 567], xy2=[697,628])

    # aple cider vinegar contour
    #bottle.label_contour = gen_square_contour(xy1=[489,439], xy2=[615,771])

    # Conicol flask contour
    #bottle.label_contour = gen_square_contour(xy1=[663,647], xy2=[704,836])






    ##____________Get Difference Image__________________##
    #Apply subtraction, mono-channel and Gaussian Blur
    bottle.diff_img = get_difference_img(control_img, bottle_img)


    ##____________Get Threshold from Difference_________##
    bottle.threshold_bottle_img = find_thresh_bottle(bottle.diff_img)


    ##____________Get Contour from Threshold_________##
    bottle_contours = find_thresh_morph_canny(bottle, itera=45, c_thresh_1=12, c_thresh_2=22)
    # For clear vessels:
    bottle_contours = find_thresh_morph_canny(bottle, itera=45, c_thresh_1=12, c_thresh_2=28)
    
    #bottle_contours = find_thresh_morph_canny(bottle, itera=15, c_thresh_1=16, c_thresh_2=60)
    bottle.raw_contour = bottle_contours[0] # extract largest contour
    
    ##_______________Bottle neck removal_____________##
    bottle = remove_neck_from_bottle_contour(bottle)


    ##____________Create bottle-only mask for fluid level processing_________##
    bottle = create_bottle_mask(bottle)


    ##____________Determine water level_________##
    #bottle = determine_fluid_level(bottle)
    #bottle = watershread_fluid_level_detect(bottle) #still used for processing algorithms

    
    lvshred_img = bottle.contour_processed_bottle_img_mono
    lvshred_bottle_contour = bottle.processed_contour
    fluid_lv, confidence, bottle = levelshred(lvshred_img, lvshred_bottle_contour, label_contour=bottle.label_contour, penalty_padding=[0.1, 0.2], kernel=5, savgol_size=-1, gt_window_size=10, threshold=1.5, bottle_debug_obj=bottle)
    
    
    print('################################\n')
    print("Lv_shred level: {}".format(fluid_lv))
    print("Lv_shred confidence: {:.1f}%\n".format(confidence))
    print('################################')


    bottle.confidence = confidence
    ##_____________Calculate volumes____________##

    bottle.volume_ml = px_area_to_standard_volume(bottle.processed_contour, PX_MM_CONVERSION, bottle_shape='c', standardise_ml = True)
    
    if (bottle.has_fluid_level):
        print('calculating fluid volume')
        bottle.fluid_level_processed_contour = extract_subcontour(bottle.processed_contour, y_cutoff=bottle.fluid_level_y)
        bottle.fluid_ml = px_area_to_standard_volume(bottle.fluid_level_processed_contour, PX_MM_CONVERSION, bottle_shape='c', standardise_ml = False)
    
    
    print(bottle.volume_ml)
    
    


    ##_____________Render results____________##
    #NOTE: Manual cropping of bottle contour
    manual_y_cutoff = 0
    if manual_y_cutoff != 0:
        bottle.processed_contour = extract_subcontour(bottle.processed_contour, y_cutoff=manual_y_cutoff, add_bounding_pts=True, cut_below=False)
        bottle.fluid_level_processed_contour = extract_subcontour(bottle.fluid_level_processed_contour, y_cutoff=manual_y_cutoff, add_bounding_pts=True, cut_below=False)
    actual_fluid_lv = 595
    #bottle.upper_y_watershread_lim = 878


    bottle = draw_bottle_and_fluid_contours_and_labels_levelshred_w_error(bottle, actual_fluid_lv)
    plot_cv_results_label_removal(bottle)
    plot_cv_results_final_res_only(bottle)
    
    #cv2.imshow("bottle", bottle_img)
    #cv2.waitKey(0)



