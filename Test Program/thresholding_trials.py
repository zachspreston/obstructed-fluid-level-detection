import cv2
import numpy as np
import imutils
import math
from skimage.metrics import structural_similarity as ssim
import random as rng
import matplotlib.pyplot as plt

from contour_class import Bottle
from process_raw_diff import get_difference_img



def find_bottle_diff_img(control_img, bottle_img):
    # convert the images to grayscale
    blw_control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2GRAY)
    blw_bottle_img = cv2.cvtColor(bottle_img, cv2.COLOR_BGR2GRAY)

    # smooth image using 7x7 pixel gaussian kernel to remove a bit of noise
    #blw_bottle_img = cv2.GaussianBlur(blw_bottle_img, (7, 7), 0)

    diff_img = cv2.subtract(blw_control_img, blw_bottle_img)
    diff_img = cv2.bitwise_not(diff_img)

    return diff_img



def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass






def canny_hull_edge_detection(bottle):
    img_original = bottle.raw_bottle_img

    blur = cv2.GaussianBlur(img_original, (3,3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    img_hue = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #img_hue[:,:,0] = 0
    # Set blue and green channels to 0
    # img[:,:,0] = 0
    # img[:,:,1] = 0
    #img_hue_only = 

    window_name = 'Canny Contour Detection'
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Canny Threshold 1', window_name, 0, 1200, nothing)
    cv2.createTrackbar('Canny Threshold 2', window_name, 0, 1200, nothing)


    while True:
        cannyThreshold1 = cv2.getTrackbarPos('Canny Threshold 1', window_name)
        cannyThreshold2 = cv2.getTrackbarPos('Canny Threshold 2', window_name)

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        bottle.thresh_bottle_img = edges
        
        # Attempt to find contours
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)


        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)


        # Draw on contours
        # Draw contours + hull results
        drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.drawContours(drawing, contours, i, color)
            cv2.drawContours(drawing, hull_list, i, color)


        # Old draw contours method
        # cv2.drawContours(img, contours, -1, (0,255,0), 2)
        # cv2.drawContours(img, contours, -1, (0,255,0), 2)


        # incase more than one difference, grab the largest one
        img_diff_areas = [cv2.contourArea(contour) for contour in contours]
        (contours_ordered, img_diff_areas) = zip(*sorted(zip(contours, img_diff_areas), key=lambda a:a[1]))
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        #c = max(contours, key = cv2.contourArea)
        cv2.fillPoly(img, pts = contours, color=(0,0,255))
        #cv2.drawContours(img, contours_ordered[0], 0, (0,0,255), 3)


        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), drawing), axis=1)

        cv2.imshow(window_name, combined)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



def canny_edge_detection_v2(bottle):
    
    img_original = bottle.raw_bottle_img
    #img_original = get_difference_img(bottle.control_img, bottle.raw_bottle_img)
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    window_name = 'Canny Contour Detection v2'
    cv2.namedWindow(window_name)
    #cv2.createTrackbar('Gaussian Blur Kernel', window_name, 0, 12, nothing)
    cv2.createTrackbar('Canny Threshold 1', window_name, 0, 200, nothing)
    cv2.createTrackbar('Canny Threshold 2', window_name, 0, 200, nothing)
    cv2.createTrackbar('Morph Iterations', window_name, 0, 40, nothing)


    while True:
        cannyThreshold1 = cv2.getTrackbarPos('Canny Threshold 1', window_name)
        cannyThreshold2 = cv2.getTrackbarPos('Canny Threshold 2', window_name)
        itera = cv2.getTrackbarPos('Morph Iterations', window_name)
        #gauss_kern = cv2.getTrackbarPos('Gaussian Blur Kernel', window_name)
        #gauss_kern = gauss_kern+1 if (gauss_kern % 2 == 0) else gauss_kern

        #blur = cv2.GaussianBlur(img_original, (9,9), 0)
        #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        #img_hue = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        bottle.thresh_bottle_img = edges

        kern=4
        kernel = np.ones((kern,kern),np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=itera)
        



        # Attempt to find contours
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Draw on contours
        cv2.drawContours(img, contours, -1, (0,255,0), 2)
        # incase more than one difference, grab the largest one

        if len(contours) == 0:
            pass
        else:
            sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    
            cv2.fillPoly(img, pts = contours, color=(0,0,255))
            # display largest contour
            cv2.fillPoly(img, pts = [sorted_contours[0]], color=(255,0,255))

        cv2.putText(img, 'GAUSSIAN KERN {}'.format(5), (5, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(img, 'CANNY {}, {}'.format(cannyThreshold1, cannyThreshold2), (5, 100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(img, 'MORPH {}, {}'.format(kern, itera), (5, 150),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), img), axis=1)

        cv2.imshow(window_name, combined)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break














def canny_edge_detection(bottle):
    img_original = bottle.raw_bottle_img

    blur = cv2.GaussianBlur(img_original, (9,9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    img_hue = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #img_hue[:,:,0] = 0
    # Set blue and green channels to 0
    # img[:,:,0] = 0
    # img[:,:,1] = 0
    #img_hue_only = 

    window_name = 'Canny Contour Detection'
    cv2.namedWindow(window_name)
    #cv2.createTrackbar('Gaussian Blur Kernel', window_name, 0, 12, nothing)
    cv2.createTrackbar('Canny Threshold 1', window_name, 0, 1200, nothing)
    cv2.createTrackbar('Canny Threshold 2', window_name, 0, 1200, nothing)
    cv2.createTrackbar('Morph Iterations', window_name, 0, 30, nothing)


    while True:
        cannyThreshold1 = cv2.getTrackbarPos('Canny Threshold 1', window_name)
        cannyThreshold2 = cv2.getTrackbarPos('Canny Threshold 2', window_name)
        itera = cv2.getTrackbarPos('Morph Iterations', window_name)

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        bottle.thresh_bottle_img = edges

        kern=4
        kernel = np.ones((kern,kern),np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=itera)
        
        # Attempt to find contours
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Draw on contours
        cv2.drawContours(img, contours, -1, (0,255,0), 2)
        # incase more than one difference, grab the largest one
        img_diff_areas = [cv2.contourArea(contour) for contour in contours]
        (contours_ordered, img_diff_areas) = zip(*sorted(zip(contours, img_diff_areas), key=lambda a:a[1]))
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        #c = max(contours, key = cv2.contourArea)
        cv2.fillPoly(img, pts = contours, color=(0,0,255))
        #cv2.drawContours(img, contours_ordered[0], 0, (0,0,255), 3)
        cv2.putText(img, 'CANNY {}, {}'.format(cannyThreshold1, cannyThreshold2), (5, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), img), axis=1)

        cv2.imshow(window_name, combined)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break





def houghP(bottle):
    img_original = bottle.raw_bottle_img

    blur = cv2.GaussianBlur(img_original, (9,9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Hough Line Transform')
    cv2.createTrackbar('Canny Threshold 1', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar('Canny Threshold 2', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar("Min Line Length", 'Hough Line Transform', 0, 100, nothing)
    cv2.createTrackbar("Max Line Gap", 'Hough Line Transform', 0, 100, nothing)

    while True:
        minLineLength = cv2.getTrackbarPos('Min Line Length', 'Hough Line Transform')
        maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Hough Line Transform')
        cannyThreshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Hough Line Transform')
        cannyThreshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Hough Line Transform')

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        # Attempt to detect straight lines in the edge detected image.
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

        # For each line that was detected, draw it on the img.
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

        cv2.imshow('Hough Line Transform', combined)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



def houghNormal(bottle):
    img_original = bottle.raw_bottle_img
    
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Hough Line Transform')
    cv2.createTrackbar('CannyThreshold1', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar('CannyThreshold2', 'Hough Line Transform', 0, 1200, nothing)
    cv2.createTrackbar("HoughThreshold", 'Hough Line Transform', 0, 200, nothing)

    while True:
        houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'Hough Line Transform')
        cannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Hough Line Transform')
        cannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Hough Line Transform')

        # Create a new copy of the original image for drawing on later.
        img = img_original.copy()
        # Use the Canny Edge Detector to find some edges.
        edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
        # Attempt to detect straight lines in the edge detected image.
        lines = cv2.HoughLines(edges, 1, np.pi/180, houghThreshold)

        # For each line that was detected, draw it on the img.
        if lines is not None:
            for line in lines:
                for rho,theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

        cv2.imshow('Hough Line Transform', combined)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break








def bottle_interact(bottle):
    window_name = 'Bottle Interaction'
    cv2.namedWindow(window_name)
    cv2.createTrackbar("threshold", window_name, 75, 255, nothing)
    cv2.createTrackbar("kernel", window_name, 5, 30, nothing)
    cv2.createTrackbar("iterations", window_name, 1, 10, nothing)

    while(True):
        # get input values
        thresh = cv2.getTrackbarPos("threshold", window_name)
        kern = cv2.getTrackbarPos("kernel", window_name)
        itera = cv2.getTrackbarPos("iterations", window_name) 
        
        # process threshold
        _,thresh_img = cv2.threshold(bottle.diff_img, thresh, 255, cv2.THRESH_BINARY_INV)
        
        # undertake kernel related operations
        kernel = np.ones((kern,kern),np.uint8) # square image kernel used for erosion
        
        #proc_img = cv2.dilate(thresh_img, kernel, iterations=itera)
        
        proc_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=itera)
        #gradient = cv2.morphologyEx(proc_img, cv2.MORPH_GRADIENT, kernel, iterations=itera)

        #proc_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        #erosion = cv2.erode(dilation,kernel,iterations = itera) # refines all edges in the binary image
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) 


        cv2.imshow(window_name, proc_img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()






def sobel_interaction(bottle):
    window_name = 'Sobel Interaction'
    cv2.namedWindow(window_name)
    cv2.createTrackbar("threshold", window_name, 75, 255, nothing)
    cv2.createTrackbar("kernel", window_name, 5, 30, nothing)
    cv2.createTrackbar("iterations", window_name, 1, 10, nothing)

    while(True):
        # get input values
        thresh = cv2.getTrackbarPos("threshold", window_name)
        kern = cv2.getTrackbarPos("kernel", window_name)
        itera = cv2.getTrackbarPos("iterations", window_name) 
        
        # process threshold
        #_,thresh_img = cv2.threshold(bottle.diff_img, thresh, 255, cv2.THRESH_BINARY_INV)
        sobelx = cv2.Sobel(bottle.diff_img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(bottle.diff_img, cv2.CV_64F, 0, 2, ksize=7, ddepth=-1)

        proc_img = sobely
        
        # undertake kernel related operations
        #kernel = np.ones((kern,kern),np.uint8) # square image kernel used for erosion
        
        #proc_img = cv2.dilate(thresh_img, kernel, iterations=itera)
        
        #proc_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=itera)
        #gradient = cv2.morphologyEx(proc_img, cv2.MORPH_GRADIENT, kernel, iterations=itera)

        #proc_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
        #erosion = cv2.erode(dilation,kernel,iterations = itera) # refines all edges in the binary image
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) 


        cv2.imshow(window_name, proc_img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()









def hsv_trials(bottle):
    window_name = 'HSV Trials'
    cv2.namedWindow(window_name)
    cv2.createTrackbar("H", window_name, 0, 255, nothing)
    cv2.createTrackbar("S", window_name, 0, 255, nothing)
    cv2.createTrackbar("V", window_name, 0, 255, nothing)

    while(True):
        # get input values
        hue = cv2.getTrackbarPos("H", window_name)
        sat = cv2.getTrackbarPos("S", window_name)
        val = cv2.getTrackbarPos("V", window_name) 
        
        # Bottle image
        img = bottle.final_leveled_bottle_img.copy()
        
        # process modified channels
        proc_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        proc_img[:,:,0] += hue
        proc_img[:,:,1] += sat
        proc_img[:,:,2] += val


        # Process Threshold
        #proc_img = proc_img[:,:,0]

        # define range of blue color in HSV
        lower_thresh = np.array([0, 150, 50])
        upper_thresh = np.array([60, 255, 255])

        mask = cv2.inRange(proc_img, lower_thresh, upper_thresh)
        res = cv2.bitwise_and(proc_img, proc_img, mask=mask)
       
        # recolour image
        img_masked = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        img_masked_grey = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
        
        ret,thresh1 = cv2.threshold(img_masked_grey,50,255,cv2.THRESH_BINARY)





        # Attempt to find contours

        contours = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # # Draw on contours
        cv2.drawContours(img, contours, -1, (0,255,0), 2)
        # # incase more than one difference, grab the largest one

        # if len(contours) == 0:
        #     pass
        # else:
        #     sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    
        #     cv2.fillPoly(img, pts = contours, color=(0,0,255))
        #     # display largest contour
        #     cv2.fillPoly(img, pts = [sorted_contours[0]], color=(255,0,255))

        # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
        #combined = np.concatenate(img_masked, img), axis=1)

        cv2.imshow(window_name, img)









        #cv2.imshow(window_name, img_render)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()







def main():
    control_img = cv2.imread("./bottle_images/no_bottle.png")
    # bottle_img = cv2.imread("./bottle_images/bottle_blank.png")
    # #bottle_img = cv2.imread("./bottle_images/bottle_whiskey.png")
    # #bottle_img = cv2.imread("./bottle_images/drink-bottle-green.png")
    # bottle_img = cv2.imread("./bottle_images/drink-bottle.png")
    # #bottle_img = cv2.imread("./bottle_images/bottle_metal.png")
    # #bottle_img = cv2.imread("./bottle_images/bottle_green_off_centre.png")
    bottle_img = cv2.imread("./bottle_images/cola.png")
    # #bottle_img = cv2.imread("./bottle_images/bottle_green_half.png")
    #bottle_img = cv2.imread("./bottle_images/water-glass-1.png")
    #bottle_img = cv2.imread("./bottle_images/bottle_2.png")

    #control_img = cv2.imread("./label_bottle_images/control_img.png")
    #bottle_img = cv2.imread("./label_bottle_images/green_bottle_v1.png")


    #control_img = cv2.imread("./live_bottle_images/control_img.png")
    #bottle_img = cv2.imread("./live_bottle_images/bottle_img.png")


    bottle = Bottle(control_img, bottle_img)
    bottle.diff_img = find_bottle_diff_img(control_img, bottle_img)
    #bottle.diff_img = bottle_img
    bottle.final_leveled_bottle_img = bottle_img



    #bottle_interact(bottle)
    #houghNormal(bottle)
    #houghP(bottle)
    #canny_hull_edge_detection(bottle)
    canny_edge_detection_v2(bottle)
    #hsv_trials(bottle)
    #cv2.imshow("bottle", bottle_img)
    #cv2.waitKey(0)
    sobel_interaction(bottle)


main()



