# video_from_webcam.py

import cv2
import os
import glob
import numpy as np
import time
import sys, getopt
from process_raw_diff import get_difference_img


def main(argv):
    #__________________set up arguments_____________________
    try:
        opts, args = getopt.getopt(argv, "m:")
        setup_mode = opts[0][1]
        
        
    except getopt.GetoptError:
        setup_mode = 'FULL'
        print('invalid input: try of form -> -m FULL OR -m BO') #BO -> bottle_only

    # Open the computers default camera
    cap = cv2.VideoCapture(0)

    ##__________________Save control image__________________##
    # save a control image after 30 frames
    if (setup_mode == 'FULL'):

        cam_background=0
        for i in range(20):
            print('Sampling background...')
            ret,cam_background=cap.read()

        cv2.imwrite('./live_bottle_images/control_img.png', cam_background)
        print('Successfully saved control image!')
    else:

        cam_background = cv2.imread("./live_bottle_images/control_img.png")
        print('Setup mode: using previous background setup')

    print('...................')
    time.sleep(1)
    print('Ready for bottle placement:')
    print('Press \'q\' to save layout')




    ##__________________Save bottle image___________________##
    while True:
        
        ret, frame = cap.read()  # Read an image from the frame.
        #frame=np.flip(cam_background, axis=1)
        
        # determine difference image
        diff_img = get_difference_img(cam_background, frame)
        
        cv2.imshow('frame', diff_img)  # Show the image on the display.
        #cv2.imshow('diff', get_difference_img(cam_background, frame))
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the script when q is pressed.
            # Save the frame
            if (setup_mode != "VIEW"):
                cv2.imwrite('./live_bottle_images/bottle_img.png', frame)
                print('...................')
                print('Successfully saved bottle image!')
            print('Setup complete')
            break

    # Release the camera device and close the GUI.
    cap.release()
    cv2.destroyAllWindows()

main(sys.argv[1:])