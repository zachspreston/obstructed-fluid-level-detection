import cv2
import numpy as np
import imutils
import math




#____________________________Area calcs____________________________#

# Func to convert contour area into volume
#bottle shape options
    #   c -> cylindrical
    #   r -> rectangular
def px_area_to_standard_volume(bottle_contour, px_to_mm_conversion, bottle_shape='c', standardise_ml = False):
    standard_volumes = [155, 200, 330, 375, 500, 750, 1000, 1250, 1500]
    
    # calc bottle area
    print('_______________')
    area_px2 = cv2.contourArea(bottle_contour) 
    (x, y, w, h) = cv2.boundingRect(bottle_contour)


    bottle_mm2 = area_px2 * px_to_mm_conversion**2
    width_mm = w * px_to_mm_conversion
    #TODO: need to discount neck from calcs
    if (bottle_shape.lower() == 'r'):
        bottle_vol_mm3 = bottle_mm2 * width_mm ** 2
    
    elif (bottle_shape.lower() == 'c'):
        bottle_vol_mm3 = bottle_mm2 * (width_mm)**2 * math.pi
    else:
        raise VolumeError('Bottle shape could not be found')
    
    bottle_vol_ml = bottle_vol_mm3 * 1e3
    print("Estimated volume: {} mL".format(bottle_vol_ml))
    
    bottle_vol_ml_standard = bottle_vol_ml
    if standardise_ml:
        for standard_vol in standard_volumes:
            if bottle_vol_ml > standard_vol:
                bottle_vol_ml_standard = standard_vol
    
    return bottle_vol_ml_standard

