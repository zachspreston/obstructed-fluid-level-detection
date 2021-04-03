import cv2
import numpy as np
import imutils
import math

from contour_class import WaterShreadColumn, extract_subcontour, extract_subcontour_w_max


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

    # WAG for where to start and end our reigon for contour processing
    alpha_y = 0.15
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
    

        # Smooth y_intensities with moving window. Note, the smoothed window will be smaller than
        y_smoothed_intensities, y_smoothed_range = window_filter(y_intensities, y_range, win_size_rad=6)
        
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
                x_penalty_offset = 15
                x_no_penalty = i-x_penalty_offset if i < contour_centre_line else i+x_penalty_offset
                is_in_contour = cv2.pointPolygonTest(bottle.processed_contour, (x_no_penalty,y_smoothed_range[j]), False)
                # pointPolygonTest returns a 1 if the point is within the contour (accountinf for the penalty offset)
                if (is_in_contour == 1):
                    viable_y_pts.append(y_smoothed_range[j])
                    # create summation of viable pts at each y
                    viable_y_range_pts_count[j] += 1
                

        watershread_results.append(WaterShreadColumn(x_coord=i, viable_y_pts=viable_y_pts))

    

    # Determine water level
    VALID_LEVEL_THRESH = 2 # this is arbitrary, needs tuning
    FILTER_OFFSET = 15 # due to window size of the gradient method
    max_viable_y_range_pts_count = max(viable_y_range_pts_count)
    fluid_level_y = lower_y_watershread_lim + viable_y_range_pts_count.index(max_viable_y_range_pts_count) + FILTER_OFFSET
    
    if (max_viable_y_range_pts_count >= VALID_LEVEL_THRESH):
        bottle.has_fluid_level = True
        # Update bottle object with new fluid level and fluid contour extract
        bottle.fluid_level_y = fluid_level_y
        bottle.fluid_level_processed_contour = extract_subcontour(bottle.processed_contour, y_cutoff=fluid_level_y)
    
    else:
        
        print('Fluid detection failed: only {} columns of sufficient threshold. {} Required'.format(max_viable_y_range_pts_count, VALID_LEVEL_THRESH))



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

    bottle.watershread_results = watershread_results
    bottle.viable_y_range_pts_count = viable_y_range_pts_count

    return bottle









##___________________________Legacy function______________________##

# def determine_fluid_level(bottle):
#     #hardcoded for this first example
#     (x, y, w, h) = cv2.boundingRect(bottle.processed_contour)
#     contour_centre_line = int(x+w/2)

#     # WAG for where to start and end our reigon for colour processing
#     alpha = 0.2
#     bottle.upper_internal_y_border = int(y+h*(1-alpha))
#     bottle.lower_internal_y_border = int(y+h*alpha)
    
#     # add bounding lines to mono image
#     cv2.line(bottle.contour_processed_bottle_img, (0, bottle.upper_internal_y_border), (bottle.img_width, bottle.upper_internal_y_border), (255, 255, 0), thickness=2)
#     cv2.line(bottle.contour_processed_bottle_img, (0, bottle.lower_internal_y_border), (bottle.img_width, bottle.lower_internal_y_border), (255, 255, 0), thickness=2)
#     cv2.line(bottle.contour_processed_bottle_img, (contour_centre_line, 0), (contour_centre_line, bottle.img_height), (255, 255, 0), thickness=2)

    
#     # Need to smooth the mono image
#     # TODO: May be better to use hue channel of HSV colourspace instead of greyscale
#     bottle.contour_processed_bottle_img_mono = cv2.GaussianBlur(bottle.contour_processed_bottle_img_mono, (5, 5), 0)

    
#     # extract center-line px values
#     #array has yvals as first index
#     #BUG: unsure why this only returns the center val. 
#     # SOLVED: I was drawing a fucking white line over it on the previous line
#     bottle.symmetry_line_mono_px_vals = bottle.contour_processed_bottle_img_mono[bottle.lower_internal_y_border:bottle.upper_internal_y_border,contour_centre_line]
#     bottle.symmetry_line_mono_y_vals = np.linspace(bottle.lower_internal_y_border, bottle.upper_internal_y_border, len(bottle.symmetry_line_mono_px_vals))
    
#     #If there is no bottle height can be found
#     px_intensities = bottle.symmetry_line_mono_px_vals
#     if (len(px_intensities) == 0):
#         print('Failed fluid level guess: {}'.format(fluid_level_y))
#         return bottle

#     ##__________________Actual fluid level determination___________________##
#     # Apply smoothing via moving window

    
    
    
    
    
#     # Arbitrary decision
#     px_trigger_tolerance = 10 #num of shades darker to count as differnece
#     fluid_level_y = 0
   
#     lower_px_intensity = px_intensities[0]
#     for i in range(0,len(px_intensities)):
#         curr_intensity = px_intensities[i]
#         if curr_intensity + px_trigger_tolerance < lower_px_intensity:
#             fluid_level_y = int(bottle.symmetry_line_mono_y_vals[i])
#             lower_px_intensity = curr_intensity
#             print('Found new lv @@ {}'.format(fluid_level_y))

#         #if px_intensities

   
#    ##__________________Creation of fluid contour___________________##
#    #If fluid level couldn't be found
#     if (fluid_level_y < bottle.lower_internal_y_border):
#         print('Failed fluid level guess: {}'.format(fluid_level_y))
#         return bottle
   

#     #TODO: issue is this will only have a contour 
#     # Instead we should just draw a line

#     # create fluid level contour for our final image bounding box
#     # Generate fluid-level contour
#     fluid_level_contour = []
#     bottle_contour = [point[0] for point in bottle.processed_contour]
#     xmax = 0
#     xmin = bottle.img_width
#     for point in bottle_contour:
#         if point[1] > fluid_level_y:
#             xmax = point[0] if point[0] > xmax else xmax
#             xmin = point[0] if point[0] < xmin else xmax
#             fluid_level_contour.append([point]) 
    
#     # ensure points at the fluid level exist on the contour
#     fluid_level_contour.append([[xmin, fluid_level_y]])
#     fluid_level_contour.append([[xmax, fluid_level_y]])
    
#     # Update bottle object

#     bottle.fluid_level_y = fluid_level_y
#     bottle.fluid_level_processed_contour = np.array(fluid_level_contour)
#     bottle.has_fluid_level = True

#     return bottle