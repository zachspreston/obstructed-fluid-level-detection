  _     _______     _______ _     ____  _   _ ____  _____ ____  
 | |   | ____\ \   / / ____| |   / ___|| | | |  _ \| ____|  _ \ 
 | |   |  _|  \ \ / /|  _| | |   \___ \| |_| | |_) |  _| | | | |
 | |___| |___  \ V / | |___| |___ ___) |  _  |  _ <| |___| |_| |
 |_____|_____|  \_/  |_____|_____|____/|_| |_|_| \_\_____|____/ 
                        

Project is ongoing and under construction!
## Testing Details

Computational Setup:
Model: 2018 MacBook Pro 
RAM: 16 GB 2133 MHz
Processor: 2.7 GHz Quad-Core Intel Core i7
OS: macOS Catalina v10.15.7
IDE: Visual Studio Code v1.54.3
Camera: Logitech

Various test images may be found in the bottle_images, label_bottle_images and live_bottle_images folders
## Install Instructions
1. Create a python virtual environment with the following packages:


Project Dependencies:
Package                 Version
----------------------- ---------
asgiref                 3.3.1
click                   7.1.2
cycler                  0.10.0
decorator               4.4.2
Django                  3.1.7
dlib                    19.21.1
face-recognition        1.3.0
face-recognition-models 0.3.0
filterpy                1.4.5
imageio                 2.9.0
imutils                 0.5.4
kiwisolver              1.3.1
matplotlib              3.3.4
networkx                2.5
numpy                   1.20.1
opencv-contrib-python   4.5.1.48
pandas                  1.2.2
Pillow                  8.1.0
pip                     20.3.3
pyparsing               2.4.7
pytesseract             0.3.7
python-dateutil         2.8.1
pytz                    2021.1
PyWavelets              1.1.1
scikit-image            0.18.1
scipy                   1.6.1
setuptools              51.3.3
six                     1.15.0
sqlparse                0.4.1
tifffile                2021.3.17
wheel                   0.36.2


## How to use

levelshred(img, container_contour, label_contour=[], penalty_padding=[0.2, 0.2], savgol_size=-1, gt_window_size=15, threshold=1.5, bottle_debug_obj=None):

Parameters:
  img -> cv2 image object of target image
  container_contour -> contour of the container within the image(ie the reigon to run the Levelshred method on)
  exclusion_contour -> contour of any exclusion zones (ie labels or other reigons to exclude in processing)
  padding -> [x, y] % of the image pixels from the boundary that will be ignored / considered in the penalty reigon
  kernal -> Kernal for the 
  savgol_size -> filter size for savgol filter, -1 specifies the autonomous selection by the levelshred method
  gt_window_size -> window size for gradient test
  threshold -> intensity/pixel threshold to be recognised as a viable level
  bottle_debug_obj -> if provided an empty class, the function will return a second parameter that populates the class
  with information about the levelshred process. See countour_class for class definition

Returns:
  1. Returns the y-index of the image in which the waterlevel was detected
  2. Second val returns the certainity the algorithm has (the % of valid threshold detections)
  (3. Debug object, if specified to run in debug mode)


## Test Program
The test program may be run by running 'fluid_detect_v6.py'
Module descriptions indicate the pre-processing taken to get the image and contours prepared for the levelshred method


By Zach Preston