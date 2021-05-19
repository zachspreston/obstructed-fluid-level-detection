import cv2
import numpy as np
import imutils
import math
from scipy.signal import savgol_filter
from contour_class import WaterShreadColumn, extract_subcontour, extract_subcontour_w_max

X_PENALTY_OFFSET = 6



#_________________________Water Level Detection ORIGINAL LEGACY Methods____________________________#
def create_bottle_mask(bottle):
    # Create black background image
    mask_bg = np.zeros(bottle.raw_bottle_img.shape, np.uint8)
    mask_bg[:,:] = (0,0,0)

    # Draw filled contour to create mask
    cv2.drawContours(mask_bg, [bottle.processed_contour], 0, (255,255,255), cv2.FILLED)
    bottle.contour_processed_bottle_img = cv2.bitwise_and(mask_bg, bottle.raw_bottle_img)
    
    #convert to monotone
    bottle.contour_processed_bottle_img_mono = cv2.cvtColor(bottle.contour_processed_bottle_img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('mask',bottle.contour_processed_bottle_img_mono)
    #cv2.waitKey()

    return bottle

# Function takes in contour of container and returns an array of the width of the contour
# at each y px of the contour 
def find_contour_widths(bottle_contour):

    # define blank array to store the widths of the contour
    contour_widths = []

    #determine dimensions and position of the contour
    (x, y, w, h) = cv2.boundingRect(bottle_contour)
    contour_centre_line = int(x+w/2)
    
    # repeat process for each row
    for i in range(0,h):

        # setup 'penalty zone' around border of contour
        # the 'width' will be width of contour after negating the penalty zone
        x_penalty_offset = 6

        # use successive approximation for most efficient finding of contour bounds
        #TODO: THIS IS SO DAMN INEFFICIENT
        xmax = 0
        is_in_contour = cv2.pointPolygonTest(bottle_contour, (contour_centre_line + xmax, i + y), False)
        while (is_in_contour > 0):
            xmax+=1
            is_in_contour = cv2.pointPolygonTest(bottle_contour, (contour_centre_line + xmax, i + y), False)
        
        xmin = 0
        is_in_contour = cv2.pointPolygonTest(bottle_contour, (contour_centre_line - xmin, i + y), False)
        while (is_in_contour > 0):
            xmin+=1
            is_in_contour = cv2.pointPolygonTest(bottle_contour, (contour_centre_line - xmin, i + y), False)
            
        
        # width is double the xmax after subtracting the penalty offset zone
        curr_width = int(xmax+xmin) #- x_penalty_offset
        # we set a minimum width in the case of inversions or contour issues
        alpha_width_min = int(0.2*w)
        if curr_width < alpha_width_min:
            curr_width = alpha_width_min
        contour_widths.append(curr_width)


    return contour_widths



def mitigate_label_from_contour_widths(contour_widths, label_contour):
    pass



#_________________________Methods for Levelshred Algorithm____________________________#

#win_size_rad applies smoothing to intesnsities array
def window_filter(y_intensities, y_range, win_size_rad):
     
     # the original y range
        y_intensities_smoothed = []

        for j in range(win_size_rad, len(y_intensities)-win_size_rad):
            curr_window = y_intensities[j-win_size_rad:j+win_size_rad]
            y_intensities_smoothed.append(sum(curr_window) / len(curr_window))
        y_smoothed_range = y_range[win_size_rad:len(y_intensities)-win_size_rad]
        
        return y_intensities_smoothed, y_smoothed_range


# Version 1 for watershread method
def watershread_fluid_level_detect(bottle):

    #determine dimensions and position of the contour
    (x, y, w, h) = cv2.boundingRect(bottle.processed_contour)
    contour_centre_line = int(x+w/2)

    # determine the contour widths of the bottle
    # used for finding % of viable points relative to width of contour at that height
    bottle.contour_widths = find_contour_widths(bottle.processed_contour)

    # WAG for where to start and end our reigon for contour processing
    alpha_y = 0.20
    alpha_x = 0.1
    upper_y_watershread_lim = int(y+h*(1-alpha_y))
    lower_y_watershread_lim = int(y+h*alpha_y)
    upper_x_watershread_lim = int(x+w*(1-alpha_x))
    lower_x_watershread_lim = int(x+w*alpha_x)

    # add bounding lines to mono image
    cv2.line(bottle.contour_processed_bottle_img, (0, upper_y_watershread_lim), (bottle.img_width, upper_y_watershread_lim), (255, 255, 0), thickness=2)
    cv2.line(bottle.contour_processed_bottle_img, (0, lower_y_watershread_lim), (bottle.img_width, lower_y_watershread_lim), (255, 255, 0), thickness=2)

    cv2.line(bottle.contour_processed_bottle_img, (upper_x_watershread_lim, 0), (upper_x_watershread_lim, bottle.img_height), (255, 255, 0), thickness=2)
    cv2.line(bottle.contour_processed_bottle_img, (lower_x_watershread_lim, 0), (lower_x_watershread_lim, bottle.img_height), (255, 255, 0), thickness=2)

    # Image used for watershread process
    watershread_img = bottle.contour_processed_bottle_img_mono.copy()
    
    # Create 2D list of watershread column points
    watershread_results = []
    viable_y_range_pts_count = [0]*(upper_y_watershread_lim-lower_y_watershread_lim)
   
    
    # Create array to store 
    for i in range(lower_x_watershread_lim, upper_x_watershread_lim):
        # extract current y column to analyse
        # In extracting vals from image, dimension1 is y vals, dimension2 is x vals, dimenson3 is intensities (or array of channels if not mono)
        y_intensities = watershread_img[lower_y_watershread_lim:upper_y_watershread_lim,i]
        y_range = np.linspace(lower_y_watershread_lim, upper_y_watershread_lim, len(y_intensities))
        viable_y_pts = []
        unviable_y_pts = [] # used to store points found within the penalty zone of the container     

        # Smooth y_intensities with moving window. Note, the smoothed window will be smaller than
    
        y_smoothed_intensities = savgol_filter(y_intensities, 19, 3)
        y_smoothed_intensities = savgol_filter(y_smoothed_intensities, 19, 2)
        y_smoothed_range = y_range

        # Smoothing method 2:
        #y_smoothed_intensities, y_smoothed_range = window_filter(y_intensities, y_range, win_size_rad=6)

        # Smoothing method 3:
        #y_smoothed_intensities = cv2.GaussianBlur(watershread_img.copy(), (7,7),0)[lower_y_watershread_lim:upper_y_watershread_lim,i]

        # Run gradient test
        window_size = 15
        threshold_gradient = 1.5 #bytes of intensity per pixel

        # will check if any changed greater than threshold_gradient occur over window size of 5px
        for j in range(0, len(y_smoothed_intensities[:-window_size]), int(window_size/2)):
            curr_gradient = abs(y_smoothed_intensities[j] - y_smoothed_intensities[j+window_size])/window_size

            if (curr_gradient > threshold_gradient):
                # Only add if point is within the contour
                # the x_penalty_offset adds an offset to x to push points outside of the contour
                # if they are within x_penalty_offset pixels of the contour border (likely spurious)
                x_penalty_offset = X_PENALTY_OFFSET
                x_no_penalty = i-x_penalty_offset if i < contour_centre_line else i+x_penalty_offset
                is_in_contour = cv2.pointPolygonTest(bottle.processed_contour, (x_no_penalty,y_smoothed_range[j]), False)
                # pointPolygonTest returns a 1 if the point is within the contour (accountinf for the penalty offset)
                if (is_in_contour == 1):
                    viable_y_pts.append(y_smoothed_range[j])
                    # create summation of viable pts at each y
                    viable_y_range_pts_count[j] += 1
                else:
                    # if in penalty-zone, add to unviable points array for visualisation on plot
                    unviable_y_pts.append(y_smoothed_range[j])
                

        watershread_results.append(WaterShreadColumn(x_coord=i, viable_y_pts=viable_y_pts, unviable_y_pts=unviable_y_pts))

    # Weight each row of viable points relative to the width of the row
    viable_y_range_pts_count_weighted = [0]*len(viable_y_range_pts_count)
    for i in range(0, len(viable_y_range_pts_count)):
        viable_y_range_pts_count_weighted[i] = 100.0*(viable_y_range_pts_count[i]/bottle.contour_widths[i+(lower_x_watershread_lim-x)])
    


    # Determine water level
    VALID_LEVEL_THRESH = 2 # this is arbitrary, needs tuning
    FILTER_OFFSET = 15 # due to window size of the gradient method
    max_viable_y_range_pts_count = max(viable_y_range_pts_count)
    #fluid_level_y = lower_y_watershread_lim + viable_y_range_pts_count.index(max_viable_y_range_pts_count) + FILTER_OFFSET
    max_viable_y_range_pts_count_weighted = max(viable_y_range_pts_count_weighted)
    fluid_level_y = lower_y_watershread_lim + viable_y_range_pts_count_weighted.index(max_viable_y_range_pts_count_weighted) + FILTER_OFFSET
    
    if (max_viable_y_range_pts_count >= VALID_LEVEL_THRESH):
        bottle.has_fluid_level = True
        # Update bottle object with new fluid level and fluid contour extract
        bottle.fluid_level_y = fluid_level_y
        bottle.fluid_level_processed_contour = extract_subcontour(bottle.processed_contour, y_cutoff=fluid_level_y)
    
    else:
        
        print('Fluid detection failed: only {} columns of sufficient threshold. {} Required'.format(max_viable_y_range_pts_count_weighted, VALID_LEVEL_THRESH))







    bottle.symmetry_line_mono_px_vals = y_intensities
    bottle.symmetry_line_mono_y_vals = y_range
    
    bottle.symmetry_line_mono_px_vals_smoothed = y_smoothed_intensities
    bottle.symmetry_line_mono_px_vals_smoothed_y = np.linspace(lower_y_watershread_lim, upper_y_watershread_lim, len(y_smoothed_intensities))


    # Update bottle object
    bottle.watershed_x_range = np.linspace(lower_x_watershread_lim, upper_x_watershread_lim, len(y_smoothed_intensities))
    bottle.watershed_y_range = np.linspace(lower_y_watershread_lim, upper_y_watershread_lim, len(y_smoothed_intensities))
    
    bottle.upper_y_watershread_lim = upper_y_watershread_lim 
    bottle.lower_y_watershread_lim = lower_y_watershread_lim
    bottle.upper_x_watershread_lim = upper_x_watershread_lim
    bottle.lower_x_watershread_lim = lower_x_watershread_lim 

    # watershread results stored as array of WaterShreadColumn objects. Array index matches y-height of contour
    bottle.watershread_results = watershread_results
    bottle.viable_y_range_pts_count = viable_y_range_pts_count # index is the y-height of contour
    bottle.viable_y_range_pts_count_weighted = viable_y_range_pts_count_weighted 
    return bottle


















# ##_________________________Lecgacy Functions_____________________________##


# # Version 1 for watershread method
# def watershread_fluid_level_detect_v1(bottle):

#     #determine dimensions and position of the contour
#     (x, y, w, h) = cv2.boundingRect(bottle.processed_contour)
#     contour_centre_line = int(x+w/2)

#     # WAG for where to start and end our reigon for contour processing
#     alpha_y = 0.20
#     alpha_x = 0.1
#     upper_y_watershread_lim = int(y+h*(1-alpha_y))
#     lower_y_watershread_lim = int(y+h*alpha_y)
#     upper_x_watershread_lim = int(x+w*(1-alpha_x))
#     lower_x_watershread_lim = int(x+w*alpha_x)

#     # add bounding lines to mono image
#     cv2.line(bottle.contour_processed_bottle_img, (0, upper_y_watershread_lim), (bottle.img_width, upper_y_watershread_lim), (255, 255, 0), thickness=2)
#     cv2.line(bottle.contour_processed_bottle_img, (0, lower_y_watershread_lim), (bottle.img_width, lower_y_watershread_lim), (255, 255, 0), thickness=2)

#     cv2.line(bottle.contour_processed_bottle_img, (upper_x_watershread_lim, 0), (upper_x_watershread_lim, bottle.img_height), (255, 255, 0), thickness=2)
#     cv2.line(bottle.contour_processed_bottle_img, (lower_x_watershread_lim, 0), (lower_x_watershread_lim, bottle.img_height), (255, 255, 0), thickness=2)

#     # Image used for watershread process
#     watershread_img = bottle.contour_processed_bottle_img_mono.copy()
    
#     # Create 2D list of watershread column points
#     watershread_results = []
#     viable_y_range_pts_count = [0]*(upper_y_watershread_lim-lower_y_watershread_lim)
    
#     # Create array to store 
#     for i in range(lower_x_watershread_lim, upper_x_watershread_lim):
#         # extract current y column to analyse
#         # In extracting vals from image, dimension1 is y vals, dimension2 is x vals, dimenson3 is intensities (or array of channels if not mono)
#         y_intensities = watershread_img[lower_y_watershread_lim:upper_y_watershread_lim,i]
#         y_range = np.linspace(lower_y_watershread_lim, upper_y_watershread_lim, len(y_intensities))
#         viable_y_pts = []
#         unviable_y_pts = [] # used to store points found within the penalty zone of the container     

#         # Smooth y_intensities with moving window. Note, the smoothed window will be smaller than
#         y_smoothed_intensities, y_smoothed_range = window_filter(y_intensities, y_range, win_size_rad=6)
        
#         # Run gradient test
#         window_size = 15
#         threshold_gradient = 1.5 #bytes of intensity per pixel

#         # will check if any changed greater than threshold_gradient occur over window size of 5px
#         for j in range(0, len(y_smoothed_intensities[:-window_size]), int(window_size/2)):
#             curr_gradient = abs(y_smoothed_intensities[j] - y_smoothed_intensities[j+window_size])/window_size

#             if (curr_gradient > threshold_gradient):
#                 # Only add if point is within the contour
#                 # the x_penalty_offset adds an offset to x to push points outside of the contour
#                 # if they are within x_penalty_offset pixels of the contour border (likely spurious)
#                 x_penalty_offset = 6
#                 x_no_penalty = i-x_penalty_offset if i < contour_centre_line else i+x_penalty_offset
#                 is_in_contour = cv2.pointPolygonTest(bottle.processed_contour, (x_no_penalty,y_smoothed_range[j]), False)
#                 # pointPolygonTest returns a 1 if the point is within the contour (accountinf for the penalty offset)
#                 if (is_in_contour == 1):
#                     viable_y_pts.append(y_smoothed_range[j])
#                     # create summation of viable pts at each y
#                     viable_y_range_pts_count[j] += 1
#                 else:
#                     # if in penalty-zone, add to unviable points array for visualisation on plot
#                     unviable_y_pts.append(y_smoothed_range[j])
                

#         watershread_results.append(WaterShreadColumn(x_coord=i, viable_y_pts=viable_y_pts, unviable_y_pts=unviable_y_pts))

    

#     # Determine water level
#     VALID_LEVEL_THRESH = 2 # this is arbitrary, needs tuning
#     FILTER_OFFSET = 15 # due to window size of the gradient method
#     max_viable_y_range_pts_count = max(viable_y_range_pts_count)
#     fluid_level_y = lower_y_watershread_lim + viable_y_range_pts_count.index(max_viable_y_range_pts_count) + FILTER_OFFSET
    
#     if (max_viable_y_range_pts_count >= VALID_LEVEL_THRESH):
#         bottle.has_fluid_level = True
#         # Update bottle object with new fluid level and fluid contour extract
#         bottle.fluid_level_y = fluid_level_y
#         bottle.fluid_level_processed_contour = extract_subcontour(bottle.processed_contour, y_cutoff=fluid_level_y)
    
#     else:
        
#         print('Fluid detection failed: only {} columns of sufficient threshold. {} Required'.format(max_viable_y_range_pts_count, VALID_LEVEL_THRESH))



#     bottle.symmetry_line_mono_px_vals = y_intensities
#     bottle.symmetry_line_mono_y_vals = y_range
    
#     bottle.symmetry_line_mono_px_vals_smoothed = y_smoothed_intensities
#     bottle.symmetry_line_mono_px_vals_smoothed_y = np.linspace(lower_y_watershread_lim, upper_y_watershread_lim, len(y_smoothed_intensities))


#     # Update bottle object
#     bottle.watershed_x_range = np.linspace(lower_x_watershread_lim, upper_x_watershread_lim, len(y_smoothed_intensities))
#     bottle.watershed_y_range = np.linspace(lower_y_watershread_lim, upper_y_watershread_lim, len(y_smoothed_intensities))
    
#     bottle.upper_y_watershread_lim = upper_y_watershread_lim 
#     bottle.lower_y_watershread_lim = lower_y_watershread_lim
#     bottle.upper_x_watershread_lim = upper_x_watershread_lim
#     bottle.lower_x_watershread_lim = lower_x_watershread_lim 

#     bottle.watershread_results = watershread_results
#     bottle.viable_y_range_pts_count = viable_y_range_pts_count

#     return bottle


