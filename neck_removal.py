import cv2
import numpy as np
import imutils
import math
from contour_class import extract_subcontour


#____________________________Neck Removal____________________________#

def remove_neck_from_bottle_contour(bottle):
    NECK_GRADIENT_THRESHOLD = 80 #degrees

    #determine x and y of the image
    (x, y, w, h) = cv2.boundingRect(bottle.raw_contour)
   
    #convert numpy array into list format
    bottle_contour = [point[0] for point in bottle.raw_contour]


    #bottle_xmin = min(bottle_contour, key = lambda t: t[1])[0] 

    # setup empty indices
    target_points = [bottle_contour[0]] #stores points used in analysis (ie via the mod filter in the loop) 
    flat_points = [] # stores points where gradient to previous point is < 90 - NECK_GRADIENT_THRESHOLD
    borderline_points = []
    #iterate through points and list indexes where gradient change occurs
    
    # setup loop
    prev_point = bottle_contour[0]; i = 0; is_curr_vertical = False
    for point in bottle_contour[1:]:
        if (i % 2 == 0):
            curr_gradient = math.degrees(math.atan((point[1]*1.0 - prev_point[1]*1.0)/(point[0]*1.0 - prev_point[0]*1.0)))
            
            #print("prev: [{},{}] | curr: [{},{}] | gradient: {}".format(prev_point[0], prev_point[1], point[0], point[1], curr_gradient))
            
            # store flat points that meet gradient requriement
            if abs(curr_gradient) > NECK_GRADIENT_THRESHOLD:
                flat_points.append(point)
                if not is_curr_vertical:
                    is_curr_vertical = True
                    borderline_points.append(point)
            else:
                if is_curr_vertical:
                    is_curr_vertical = False
                    borderline_points.append(point)

            
            target_points.append(prev_point)
            #Update prev_point for next iteration
            prev_point = point
        
        i+=1

    # determine neck cutoff
    neck_cutoff = 0
    neck_border_points = []

    #TODO: fix these horrible magic numbers
    # determine by ordering by x and finding highest x point that is n contingency
    # from the output
    # search for top left neck
    # magic numbers are pixel thresholds
    if (len(borderline_points) == 0):
        print('No bottle neck found!')
        bottle.processed_contour = bottle.raw_contour
        return bottle

    prev_borderline_point = borderline_points[0]
    for point in borderline_points[1:]:
        print("prev: [{},{}] | curr: [{},{}]".format(prev_borderline_point[0], prev_borderline_point[1], point[0], point[1]))
        if ((point[0] - prev_borderline_point[0]) > 6) and (6 < (point[1] - prev_borderline_point[1]) < 150):
            # only select neck-like points
            
            #must also pass y test (if in top half of bottle)
            #and x text (must be within 80% of the centre of the bottle)
            
            if (point[1] < y + h/2):
                if ((abs(point[0] - (x+w/2))) < 0.8*(w/2)):
                    neck_border_points.append(point)
                    print("FOUND POINT {},{}".format(point[0], point[1]))

        #TODO: this line should be uncommented. Right now we are only comparing all points to the first point
        #prev_borderline_point = point

   
    if (len(neck_border_points) == 0):
        print('No bottle neck found!')
        bottle.processed_contour = bottle.raw_contour
        return bottle

    bottle.is_bottle_neck = True  
    sum = 0  
    for point in neck_border_points:
        sum += point[1]
    
    NECK_CUTOFF_PADDING = 20 # added pixels to mitigate any artifacts from neck
    neck_cutoff = sum/len(neck_border_points) + NECK_CUTOFF_PADDING

    # create neckless contour
    # neckless_contour = []
    # for point in bottle_contour:
    #     if point[1] > neck_cutoff:
    #         neckless_contour.append([point])
    #neckless_contour = np.array(neckless_contour)
    neckless_contour = extract_subcontour(bottle.raw_contour, y_cutoff=neck_cutoff, add_bounding_pts=True)
    print(neckless_contour)

    # Update bottle object
    bottle.proc_contour_points = target_points
    bottle.proc_flat_contour_points = flat_points
    bottle.proc_borderline_contour_points = borderline_points
    bottle.neck_y_coord = neck_cutoff
    #same thing for this case
    bottle.processed_contour = neckless_contour
    bottle.no_neck_contour = neckless_contour
    
    return bottle
