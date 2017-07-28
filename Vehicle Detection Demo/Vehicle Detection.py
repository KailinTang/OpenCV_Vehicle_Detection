# -*- coding: utf-8 -*-

# Calculate the center of ROI
def calculateCenter((a,b,c,d)):
    center_x = a + c/2
    center_y = b + d/2
    return (center_x, center_y)

# Judge whether the point is in a ROI
def isInSpecRoi((x,y), (a,b,c,d)):
    x_range = (a, a+c)
    y_range = (b, b+d)
    if x>x_range[0] and x<x_range[1]:
        if y>y_range[0] and y<y_range[1]:
            return True
    else:
        return False

# Filter out the result in false region (You should modify the values to gaurantee the correct result)
def isNotInCorrectPostion((a,b,c,d)):
    if b <= 360:
        return True
    elif b >=500:
        return True
    else:
        return False        
        
# Judge whether a region is completely included by another region
# Detect whether the first parameter is completely included by the second prarmeter
def isFullyInclude((a1,b1,c1,d1),(a2,b2,c2,d2)):
    param_1_center = calculateCenter((a1,b1,c1,d1))
    if not isInSpecRoi(param_1_center, (a2,b2,c2,d2)):
        return False
    param_1_x_range = (a1, a1+c1)
    param_1_y_range = (b1, b1+d1)
    param_2_x_range = (a2, a2+c2)
    param_2_y_range = (b2, b2+d2)
    if param_1_x_range[0] > param_2_x_range[0] and param_1_x_range[1] < param_2_x_range[1]:
        if param_1_y_range[0] > param_2_y_range[0] and param_1_y_range[1] < param_2_y_range[1]:
            return True
    else:
        return False

import cv2
import numpy as np

# Cascade classifier trained from the opencv_traincascade utility from OpenCV
cascade_src = 'cascade.xml'
# A short video demo
video_src = 'dataset/demo.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)


while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray,gray)
    
    # Hypothesis generation with sliding windows with multiple scales and verification using cascade classifier
    cars = car_cascade.detectMultiScale(gray, scaleFactor=2.3, minNeighbors=20, minSize=(20,20), maxSize=(200,200))
        
    for (x,y,w,h) in cars:	# Check whether every ROI satisfy the requirements
        roiImage = gray [y:y+h, x:x+w]
        
        flag_1 = False # Flag represents for whether a region is fully included by another region
        
        if isNotInCorrectPostion((x,y,w,h)):
            continue
        for (x_temp, y_temp, w_temp, h_temp) in cars:
            if isFullyInclude((x,y,w,h),(x_temp, y_temp, w_temp, h_temp)):
                flag_1 = True
                break
        if flag_1:
            continue
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)	# Mark the real cars
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
