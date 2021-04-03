import cv2
import numpy as np
import imutils
import math
import matplotlib.pyplot as plt







#____________________________Rendering Funcs____________________________#



def draw_bottle_and_fluid_contours_and_labels(bottle_obj):
    # Overlay contours into original image
    # params are base image, contour maps, which contour in list (-1 for all), colour, thickness 
    #cv2.drawContours(bottle_obj.final_leveled_bottle_img, bottle_obj.raw_contour, 0, (0,255,0), 3)
    (x, y, w, h) = cv2.boundingRect(bottle_obj.processed_contour)
    cv2.rectangle(bottle_obj.final_leveled_bottle_img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    cv2.putText(bottle_obj.final_leveled_bottle_img, '{:.2f} mL'.format(bottle_obj.volume_ml), (x+w+12, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)

    if (bottle_obj.has_fluid_level):
        # Draw fluid level contour + text
        (x, y, w, h) = cv2.boundingRect(bottle_obj.fluid_level_processed_contour)
        cv2.rectangle(bottle_obj.final_leveled_bottle_img, (x, y), (x + w, y + h), (255, 100, 0), 2)
        cv2.putText(bottle_obj.final_leveled_bottle_img, '{:.2f} mL'.format(bottle_obj.fluid_ml), (x+w+12, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,0), 4)

        # Add caption
        percent_full = (1.0 - (bottle_obj.volume_ml - bottle_obj.fluid_ml)/bottle_obj.volume_ml)*100
        cv2.putText(bottle_obj.final_leveled_bottle_img, 'FLUID LV: {:.1f}%'.format(percent_full), (int(bottle_obj.img_width*0.5), int(bottle_obj.img_height*0.98)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4)

    else:
        cv2.putText(bottle_obj.final_leveled_bottle_img, 'FLUID LV NOT FOUND', (int(bottle_obj.img_width*0.5), int(bottle_obj.img_height*0.98)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4)



    return bottle_obj


#____________________________Plotting Funcs__________________________#

# Prints initial 2x5 plots
def plot_cv_results_original(bottle):

    img = bottle.raw_bottle_img

    # determine metadata
    img_height, img_width = bottle.img_height, bottle.img_width
    
    num_rows, num_cols = 2, 5

    fig, ax = plt.subplots(figsize=(12.8, 7.5))
    fig.tight_layout()
    
    # Plot original image
    plt.subplot(num_rows, num_cols,1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(bottle.raw_bottle_img.copy(), cv2.COLOR_BGR2RGB))
    plt.axis('off')
   
    # Plot difference image
    plt.subplot(num_rows, num_cols,2)
    plt.title('Difference Extraction')
    plt.imshow(bottle.diff_img, cmap = 'Greys')
    plt.axis('off')

    # Plot canny threshold
    plt.subplot(num_rows, num_cols,3)
    plt.title('Canny-Based Threshold')
    plt.imshow(bottle.canny_bottle_img, cmap = 'Greys')
    plt.axis('off')

    # Plot threshold
    plt.subplot(num_rows, num_cols,4)
    plt.title('Morphology Opening')
    plt.imshow(bottle.threshold_bottle_img, cmap = 'Greys')
    plt.axis('off')


    # Plot contours 
    plt.subplot(num_rows, num_cols,5)
    plt.title('Determined Contour')

    x = [point[0][0] for point in bottle.raw_contour]
    y = [point[0][1] for point in bottle.raw_contour] 

    plt.plot(x,y, color='green')
    plt.plot(x,y, marker = 'x', linewidth=0, color='red', markerfacecolor='red', markersize=5)
    
    plt.xlim([0, img_width])
    plt.ylim([img_height, 0])
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.axis('off')


    # Plot processed contours
    plt.subplot(num_rows, num_cols,6)
    plt.title('Neck Determination')
    
    if (bottle.is_bottle_neck):
        # print test points
        x = [point[0] for point in bottle.proc_contour_points]
        y = [point[1] for point in bottle.proc_contour_points] 
        x_flat = [point[0] for point in bottle.proc_flat_contour_points]
        y_flat = [point[1] for point in bottle.proc_flat_contour_points] 
        x_borderline = [point[0] for point in bottle.proc_borderline_contour_points]
        y_borderline = [point[1] for point in bottle.proc_borderline_contour_points]

        plt.plot(x,y, marker = 'x', linewidth=0, color='blue', markerfacecolor='blue', markersize=5)
        plt.plot(x_flat,y_flat, marker = 'x', linewidth=0, color='red', markerfacecolor='red', markersize=5)
        plt.plot(x_borderline,y_borderline, marker = 'x', linewidth=0, color='green', markerfacecolor='green', markersize=5)
        plt.plot([0, bottle.img_width], [bottle.neck_y_coord, bottle.neck_y_coord], color='red')
        plt.xlim([0, bottle.img_width])
        plt.ylim([bottle.img_height, 0])
        
        plt.gca().set_aspect('equal', adjustable='box')
    else:
        blank_img = bottle.control_img.copy()
        cv2.putText(blank_img, 'BOTTLE HAS NO SLENDER NECK', (int(bottle.img_width*0.08), int(bottle.img_height*0.5)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
        plt.imshow(blank_img, cmap = 'Greys')



    # Plot bottle mask
    plt.subplot(num_rows, num_cols, 7)
    plt.title('Applied Contour Mask')
    plt.imshow(cv2.cvtColor(bottle.contour_processed_bottle_img.copy(), cv2.COLOR_BGR2RGB), origin='upper')
    plt.axis('off')


    # Plot line-of-symmetry intensity vals
    plt.subplot(num_rows, num_cols, 8)
    plt.title('Int Vertical Px Intensity')
    y = bottle.symmetry_line_mono_y_vals
    x = bottle.symmetry_line_mono_px_vals
    plt.plot(x,y, marker = 'x', linewidth=3, color='red', markerfacecolor='red', markersize=5)
    plt.ylim([img_height, 0])
    #plt.axis('off')


    # Plot estimated water level
    plt.subplot(num_rows, num_cols, 9)
    plt.title('Fluid Level Estimation')
    cv2.line(bottle.contour_processed_bottle_img_mono, (0, bottle.fluid_level_y), (bottle.img_width, bottle.fluid_level_y), (255, 255, 0), thickness=2)
    if (bottle.has_fluid_level):
        cv2.putText(bottle.contour_processed_bottle_img_mono, 'Lv@ y={}'.format(bottle.fluid_level_y), (int(bottle.img_width*0.65), bottle.fluid_level_y-5),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    plt.imshow(bottle.contour_processed_bottle_img_mono, origin='upper')
    plt.axis('off')

    # Plot final image
    plt.subplot(num_rows, num_cols,10)
    plt.title('Fluid Volume Prediction')
    plt.imshow(cv2.cvtColor(bottle.final_leveled_bottle_img.copy(), cv2.COLOR_BGR2RGB), origin='upper')
    #plt.axis('off')
    plt.show()









# Prints initial 2x5 plots
def plot_cv_results_label_removal(bottle):

    img = bottle.raw_bottle_img

    # determine metadata
    img_height, img_width = bottle.img_height, bottle.img_width
    
    num_rows, num_cols = 2, 5

    fig, ax = plt.subplots(figsize=(12.8, 7.5))
    fig.tight_layout()


    
    # Plot original image
    # plt.subplot(num_rows, num_cols,1)
    # plt.title('Original Image')
    # plt.imshow(cv2.cvtColor(bottle.raw_bottle_img.copy(), cv2.COLOR_BGR2RGB))
    # plt.axis('off')
   


    # Plot difference image
    plt.subplot(num_rows, num_cols,1)
    plt.title('Difference Extraction')
    plt.imshow(bottle.diff_img, cmap = 'Greys')
    plt.axis('off')





    # Plot canny threshold
    plt.subplot(num_rows, num_cols,2)
    plt.title('Canny-Based Threshold')
    plt.imshow(bottle.canny_bottle_img, cmap = 'Greys')
    plt.axis('off')



    # Plot threshold
    plt.subplot(num_rows, num_cols,3)
    plt.title('Morphology Opening')
    plt.imshow(bottle.threshold_bottle_img, cmap = 'Greys')
    plt.axis('off')




    # Plot contours 
    plt.subplot(num_rows, num_cols,4)
    plt.title('Determined Contour')

    x = [point[0][0] for point in bottle.raw_contour]
    y = [point[0][1] for point in bottle.raw_contour] 

    plt.plot(x,y, color='green')
    plt.plot(x,y, marker = 'x', linewidth=0, color='red', markerfacecolor='red', markersize=5)
    
    plt.xlim([0, img_width])
    plt.ylim([img_height, 0])
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.axis('off')


    # Plot processed contours
    # plt.subplot(num_rows, num_cols,6)
    # plt.title('Neck Determination')
    
    # if (bottle.is_bottle_neck):
    #     # print test points
    #     x = [point[0] for point in bottle.proc_contour_points]
    #     y = [point[1] for point in bottle.proc_contour_points] 
    #     x_flat = [point[0] for point in bottle.proc_flat_contour_points]
    #     y_flat = [point[1] for point in bottle.proc_flat_contour_points] 
    #     x_borderline = [point[0] for point in bottle.proc_borderline_contour_points]
    #     y_borderline = [point[1] for point in bottle.proc_borderline_contour_points]

    #     plt.plot(x,y, marker = 'x', linewidth=0, color='blue', markerfacecolor='blue', markersize=5)
    #     plt.plot(x_flat,y_flat, marker = 'x', linewidth=0, color='red', markerfacecolor='red', markersize=5)
    #     plt.plot(x_borderline,y_borderline, marker = 'x', linewidth=0, color='green', markerfacecolor='green', markersize=5)
    #     plt.plot([0, bottle.img_width], [bottle.neck_y_coord, bottle.neck_y_coord], color='red')
    #     plt.xlim([0, bottle.img_width])
    #     plt.ylim([bottle.img_height, 0])
        
    #     plt.gca().set_aspect('equal', adjustable='box')
    # else:
    #     blank_img = bottle.control_img.copy()
    #     cv2.putText(blank_img, 'BOTTLE HAS NO SLENDER NECK', (int(bottle.img_width*0.08), int(bottle.img_height*0.5)),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
    #     plt.imshow(blank_img, cmap = 'Greys')



    # Plot bottle mask
    plt.subplot(num_rows, num_cols, 5)
    plt.title('Applied Contour Mask')
    plt.imshow(cv2.cvtColor(bottle.contour_processed_bottle_img.copy(), cv2.COLOR_BGR2RGB), origin='upper')
    plt.axis('off')


    # Plot line-of-symmetry intensity vals
    plt.subplot(num_rows, num_cols, 6)
    plt.title('Smoothed Vertical Px Intensity')
    y = bottle.symmetry_line_mono_y_vals
    x = bottle.symmetry_line_mono_px_vals

    plt.plot(x,y, marker = 'x', linewidth=3, color='red', markerfacecolor='red', markersize=5)
    plt.plot(bottle.symmetry_line_mono_px_vals_smoothed, bottle.symmetry_line_mono_px_vals_smoothed_y, marker = 'x', linewidth=3, color='green', markerfacecolor='green', markersize=5)
    plt.ylim([img_height, 0])
    #plt.axis('off')

    # plot total px histogram
    plt.subplot(num_rows, num_cols, 7)
    plt.title('Net Vertical Px Histogram')
    plt.plot(bottle.viable_y_range_pts_count, bottle.symmetry_line_mono_y_vals, marker = 'x', linewidth=0, color='blue', markerfacecolor='green', markersize=5)
    plt.ylim([img_height, 0])



    # Plot line-of-symmetry intensity vals
    plt.subplot(num_rows, num_cols, 8)
    plt.title('Watershread Points')
    for col in bottle.watershread_results:
        plt.plot([col.x]*len(col.viable_y_pts), col.viable_y_pts, marker = 'x', linewidth=0, color='red', markerfacecolor='red', markersize=5)


    
    plt.ylim([img_height, 0])
    plt.xlim([0, img_width])





    # Plot estimated water level
    plt.subplot(num_rows, num_cols, 9)
    plt.title('Fluid Level Estimation')
    cv2.line(bottle.contour_processed_bottle_img_mono, (0, bottle.fluid_level_y), (bottle.img_width, bottle.fluid_level_y), (255, 255, 0), thickness=2)
    if (bottle.has_fluid_level):
        cv2.putText(bottle.contour_processed_bottle_img_mono, 'Lv@ y={}'.format(bottle.fluid_level_y), (int(bottle.img_width*0.65), bottle.fluid_level_y-5),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    plt.imshow(bottle.contour_processed_bottle_img_mono, origin='upper')
    plt.axis('off')

    # Plot final image
    plt.subplot(num_rows, num_cols,10)
    plt.title('Fluid Volume Prediction')
    plt.imshow(cv2.cvtColor(bottle.final_leveled_bottle_img.copy(), cv2.COLOR_BGR2RGB), origin='upper')
    #plt.axis('off')
    plt.show()
