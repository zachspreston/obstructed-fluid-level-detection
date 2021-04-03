#  Authors:        Zach Preston        zp2 
#
#  Date created: 24th Mar 2021
#  Date Last Modified: 
#
#  Module Description:
#  Classes for passing contour information across program
import cv2
import numpy as np

class Bottle:
    # Initialises relevant parameters
    def __init__(self, control_img, raw_img):
        

        # image properties
        self.control_img = control_img
        self.raw_bottle_img = raw_img
        self.diff_img = None
        self.threshold_bottle_img = None
        self.canny_bottle_img = None
        self.img_height, self.img_width = raw_img.shape[:2]
        
        self.contour_processed_bottle_img = None
        self.contour_processed_bottle_img_mono = None
        self.final_leveled_bottle_img = raw_img.copy()



        # contour properties
        self.raw_contour = None
        self.no_neck_contour = None
        self.processed_contour = None
        self.fluid_level_processed_contour = None
        self.contour_width = 0
        self.contour_height = 0

        self.is_bottle_neck = False
        self.proc_contour_points = []
        self.proc_flat_contour_points = []
        self.proc_borderline_contour_points = []
        self.neck_y_coord = 0

        # label properties 

        # fluid level properties
        self.upper_internal_y_border = 0
        self.lower_internal_y_border = 0
        self.has_fluid_level = False

        self.symmetry_line_mono_y_vals = [] #stores monotone px values for center line used for level detection
        self.symmetry_line_mono_px_vals = [] #stores monotone px values for center line used for level detection
        self.fluid_level_y = 0
   
   
        # processed properties
        self.bottle_shape = 'c'
        self.volume_ml = 0
        self.fluid_ml = 0


        # constants
        self.standard_volumes = [200, 330, 375, 500, 750, 1000, 1250, 1500]

    # String output of class
    def __str__(self):
        return self.raw_contour




# Class to store info on a single column  used in watershread analysis
class WaterShreadColumn:
    def __init__(self, x_coord=0, viable_y_pts=[]):
        self.x = x_coord
        self.viable_y_pts = viable_y_pts






# Creates a new contour object that returns the provided contour
# with all points above (cut_below=False), or points below (cut_below=True)
# a given y_cutoff
# This method will generate new points at the cut to ensure the contour
# is cleanly cut at that point
 
def extract_subcontour(contour, y_cutoff, cut_below=True, add_bounding_pts=True):
    # create fluid level contour for our final image bounding box
    # Generate fluid-level contour
    new_contour = []
    current_contour = [point[0] for point in contour]
    #xmax = 0
    #xmin = 99999 # an arbitrarily large number. Saves processing time as this will be quickly overwritten
    
    (x, y, w, h) = cv2.boundingRect(contour)
    x_line_of_symmetry = int(x + w/2)
    
    closest_pt = [9999,9999]
    # running the cut_below check now prevents the check on all iterations
    if cut_below:
        for point in current_contour:
            if point[1] >= y_cutoff:
                #xmax = point[0] if point[0] > xmax else xmax
                #xmin = point[0] if point[0] < xmin else xmax
                closest_pt = point if (abs(point[1] - y_cutoff) < abs(closest_pt[1] - y_cutoff)) else closest_pt
                new_contour.append([point]) 
    else:
        for point in current_contour:
            if point[1] < y_cutoff:
                #xmax = point[0] if point[0] > xmax else xmax
                #xmin = point[0] if point[0] < xmin else xmax
                closest_pt = point if (abs(point[1] - y_cutoff) < abs(closest_pt[1] - y_cutoff)) else closest_pt
                new_contour.append([point]) 

    # ensure points at the fluid level exist on the contour
    if (add_bounding_pts):
        x1 = point[0]
        x2 = x1 + 2*(x_line_of_symmetry-x1) if (x1 < x_line_of_symmetry) else x1 - 2*(x1-x_line_of_symmetry)
        print('closest pt x = {},{}'.format(x1,x2))
        new_contour.append([[int(x1), int(y_cutoff)]])
        new_contour.append([[int(x2), int(y_cutoff)]])
    
    # ensure contour returned as a NumPy array
    return np.array(new_contour)
    



def extract_subcontour_w_max(contour, y_cutoff, cut_below=True, add_bounding_pts=True):
    # create fluid level contour for our final image bounding box
    # Generate fluid-level contour
    new_contour = []
    current_contour = [point[0] for point in contour]
    xmax = 0
    xmin = 99999 # an arbitrarily large number. Saves processing time as this will be quickly overwritten
    
    # running the cut_below check now prevents the check on all iterations
    if cut_below:
        for point in current_contour:
            if point[1] >= y_cutoff:
                xmax = point[0] if point[0] > xmax else xmax
                xmin = point[0] if point[0] < xmin else xmax
                new_contour.append([point]) 
    else:
        for point in current_contour:
            if point[1] < y_cutoff:
                xmax = point[0] if point[0] > xmax else xmax
                xmin = point[0] if point[0] < xmin else xmax
                new_contour.append([point]) 

    # ensure points at the fluid level exist on the contour
    if (add_bounding_pts):
        new_contour.append([[xmin, y_cutoff]])
        new_contour.append([[xmax, y_cutoff]])
    
    # ensure contour returned as a NumPy array
    return np.array(new_contour)
    