
# Module for:     t h e   L e v e l   s h r e d   m e t h o d

import cv2
import numpy as np
import imutils
import math
import statistics
from scipy.signal import savgol_filter



# Class to store info on a single column  used in Levelshred analysis
class LevelShredColumn:
    def __init__(self, x_coord=0, y_indicies=[], viable_y_pts=[], unviable_y_pts=[], contour_centre_line=0):
        # Column metadata
        self.x = x_coord
        self.y_indicies = y_indicies
        self.px_intensities = []#self.y # same property, just different naming
        self.contour_centre_line = contour_centre_line

        # Smoothed column metadata
        self.smoothed_px_intensities = []
        self.smoothed_range = []

        # Stores number the indices of viable points in this given column
        self.viable_y_pts = viable_y_pts

        # used to store points found within the penalty zone of the container
        # no logical utility, but useful in visualisation + debugging
        self.unviable_y_pts = unviable_y_pts 



# Function takes in contour of container and returns an array of the width of the contour
# at each y px of the contour 
# the 'width' will be width of contour after negating the penalty zone
def find_contour_widths(bottle_contour, x_penalty_offset=0):

    # define blank array to store the widths of the contour
    contour_widths = []

    #determine dimensions and position of the contour
    (x, y, w, h) = cv2.boundingRect(bottle_contour)
    contour_centre_line = int(x+w/2)
    
    # repeat process for each row
    for i in range(0,h):

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


# Method to execute the gradient test
#TODO: This can be made cleaner using median filter and differentiating the function
def gradient_test(lvshred_column, container_contour, running_viable_y_pts_tally, label_contour=[], gradient_threshold=1.5, window_size=15, x_penalty_offset=0):
    
    viable_y_pts=[]
    unviable_y_pts=[]
    i = lvshred_column.x
    
    for j in range(0, len(lvshred_column.smoothed_px_intensities[:-window_size]), int(window_size/2)):
        
        curr_gradient = abs(lvshred_column.smoothed_px_intensities[j] - lvshred_column.smoothed_px_intensities[j+window_size])/window_size
        curr_y_index = j + int(window_size/2) # as we averaging over a window
        if (curr_gradient > gradient_threshold):
            # Only add if point is within the contour
            # the x_penalty_offset adds an offset to x to push points outside of the contour
            # if they are within x_penalty_offset pixels of the contour border (likely spurious)
            x_no_penalty = i-x_penalty_offset if i < lvshred_column.contour_centre_line else i+x_penalty_offset
            is_in_contour = cv2.pointPolygonTest(container_contour, (x_no_penalty,lvshred_column.y_indicies[curr_y_index]), False)
            
            if (label_contour != []):
                is_in_label_contour = cv2.pointPolygonTest(label_contour[0], (i,lvshred_column.y_indicies[curr_y_index]), False)

            
            # pointPolygonTest returns a 1 if the point is within the contour (accountinf for the penalty offset)
            if (is_in_contour == 1 and is_in_label_contour != 1):
                # Add current index to array
                viable_y_pts.append(lvshred_column.y_indicies[curr_y_index])
                # ammend running talley 
                running_viable_y_pts_tally[curr_y_index] += 1
            else:
                # if in penalty-zone, add to unviable points array for visualisation on plot
                unviable_y_pts.append(lvshred_column.y_indicies[curr_y_index])


    return viable_y_pts, running_viable_y_pts_tally, unviable_y_pts





# Parameters:
#   padding -> [x, y] % of the image pixels from the boundary that will be ignored / considered in the penalty reigon
#   kernal ->
#   savgol_size -> filter size for savgol filter
#   gt_window_size -> window size for gradient test

# Returns:
#   1. Returns the y-index of the image in which the waterlevel was detected
#   2. Second val returns the certainity the algorithm has (the % of valid threshold detections)

def levelshred(img, container_contour, label_contour=[], penalty_padding=[0.2, 0.2], kernel=5, savgol_size=-1, gt_window_size=15, threshold=1.5, bottle_debug_obj=None):

    # STEP 1.
    # If the image is colour, convert it to black & white
    
    # Apply gaussian blur
    img = cv2.GaussianBlur(img, (kernel, kernel), 0)

    # STEP 2.
    # Mask image using the contour


    # STEP 3.
    # Determine image & contour meta-data

    # Find image meta-data
    (x, y, w, h) = cv2.boundingRect(container_contour)
    contour_centre_line = int(x+w/2)

    # Define reigon of interest (lower, upper) based on penalty paddings
    lvshred_x_lim = [int(x+w*(penalty_padding[0])),int(x+w*(1-penalty_padding[0]))]
    lvshred_y_lim = [int(y+h*(penalty_padding[1])),int(y+h*(1-penalty_padding[1]))]
   
    # Stores y indices within lvshred reigon
    lvshred_y_indices = list(range(lvshred_y_lim[0],lvshred_y_lim[1]))
   
    # Determine the width of each row in the bottle contour
    # index 0 is the lowest y-index row within the contour
    container_contour_widths = find_contour_widths(container_contour, x_penalty_offset=6)

    # STEP 4.
    # Setup levelshred column objects for the mode specifed by the input parameters

    # Initialise the viable points tally. Each index stores the number of viable points
    # recorded for each column
    viable_level_tally = [0]*(lvshred_y_lim[1]-lvshred_y_lim[0])

    # Initialise blank array to store lvshred object for each pixel column in reigon of interest
    lvshred_columns = [] 


    # STEP 5.
    # Iterate over each column in the reigon of interest and setup current column for analysis
    for i in range(lvshred_x_lim[0], lvshred_x_lim[1]):
        
        # Setup lvshred object to store all info on current level
        curr_col = LevelShredColumn(x_coord=i, y_indicies=lvshred_y_indices, contour_centre_line=contour_centre_line)
        # Extract column associated to this iteration
        curr_col.px_intensities = img[lvshred_y_lim[0]:lvshred_y_lim[1],i]

        # STEP 5.2
        # Smooth column using a Savgol filter. Note, cumsum may be faster alternative
        if(savgol_size == -1):
            savgol_size = 19 #TODO: have -1 autonomously choose size based on image
        
        curr_col.smoothed_px_intensities = savgol_filter(curr_col.px_intensities, savgol_size, 3)
        curr_col.smoothed_px_intensities = savgol_filter(curr_col.smoothed_px_intensities, savgol_size, 2) # Run second pass for second order polynomial

        # STEP 6
        # Run gradient test
        curr_col.viable_y_pts, viable_level_tally, curr_col.unviable_y_pts = gradient_test(curr_col, container_contour, viable_level_tally, label_contour=label_contour, gradient_threshold=1.5, window_size=15, x_penalty_offset=5)
        lvshred_columns.append(curr_col)

    # STEP 7.
    # Calculated weighted viable points tally 

    # If a label contour was given:
    if (len(label_contour) != 0):
        pass #implement mitigation for now
    
    # Stores the number of viable points as a % of points relative to the width of the row
    weighted_viable_level_tally = [0]*len(viable_level_tally)
    for i in range(0, len(viable_level_tally)):
        # find portion of points relative to pixel width of column
        # the index i+(lvshred_x_lim[0]-x) will make index i equivilent to the lowest index in the contour
        # that is offset by the lvshred x_limit (x is used as lvshred x limit is in terms of the image itself)
        weighted_viable_level_tally[i] = 100.0*(viable_level_tally[i]/container_contour_widths[i+(lvshred_x_lim[0]-x)])

    # Secondly, we weight this tally using a modified gaussian distrbution
    # This is to penalise high-detection rates in lower & upper reigons due
    # to the likelhood of spurious light effects at these points
    #TODO


    # STEP 8.
    # Find the estimated level by finding the max weighted val

    max_weighted_viable_level_tally = max(weighted_viable_level_tally)
    fluid_level_y = lvshred_y_lim[0] + weighted_viable_level_tally.index(max_weighted_viable_level_tally)
    
    lvshred_confidence = max_weighted_viable_level_tally # confidence is outputted as the portion of points to the width of the contour




    
    if (bottle_debug_obj==None):
        return fluid_level_y, lvshred_confidence
    
    
    # UPDATE DEBUG VISUALISATION VARIABLES
    
    else:
        print('Lvshred debug mode:')
        # Set column of interest for visualisation
        curr_col = curr_col


        bottle_debug_obj.symmetry_line_mono_px_vals = curr_col.px_intensities
        bottle_debug_obj.symmetry_line_mono_y_vals = curr_col.y_indicies
        
        bottle_debug_obj.symmetry_line_mono_px_vals_smoothed = curr_col.smoothed_px_intensities
        bottle_debug_obj.symmetry_line_mono_px_vals_smoothed_y = curr_col.y_indicies


        # Update bottle object
        bottle_debug_obj.upper_y_watershread_lim = lvshred_y_lim[1]
        bottle_debug_obj.lower_y_watershread_lim = lvshred_y_lim[0]
        bottle_debug_obj.upper_x_watershread_lim = lvshred_x_lim[1]
        bottle_debug_obj.lower_x_watershread_lim = lvshred_x_lim[0] 

        # watershread results stored as array of WaterShreadColumn objects. Array index matches y-height of contour
        bottle_debug_obj.watershread_results = lvshred_columns
        bottle_debug_obj.viable_y_range_pts_count = viable_level_tally # index is the y-height of contour
        bottle_debug_obj.viable_y_range_pts_count_weighted = weighted_viable_level_tally 

        bottle_debug_obj.has_fluid_level = True
        # Update bottle object with new fluid level and fluid contour extract
        bottle_debug_obj.fluid_level_y = fluid_level_y


        return fluid_level_y, lvshred_confidence, bottle_debug_obj










