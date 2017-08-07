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

# Region separation for unnecessary detection        
def isNotInCorrectPostion((a,b,c,d)):
    if b <= 360:
        return True
    elif b >=500:
        return True
    else:
        return False
        
# Region separation for unnecessary matching          
def inCorrectRegion(x,y):
    if x > 575 and x < 675:
        if y > 350 and y < 450:
            return True
        else:
            return False
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

# Compare the size of the two ROIs, return true for parameter 1 is bigger than parameter 2 or rather return false
def roiAreaCompare((a1,b1,c1,d1), (a2,b2,c2,d2)):
    area_1 = c1 * d1
    area_2 = c2 * d2
    if area_1 >= area_2:
        return True
    else:
        return False

import cv2
import numpy as np

objectCar = cv2.imread('target.jpg',0)    # Target vehicle to match
cascade_src = 'cascade.xml'
video_src = 'dataset/Demo.mp4'

sift = cv2.SIFT()
bf = cv2.BFMatcher()
ObjKp, ObjDes = sift.detectAndCompute(objectCar,None)    # SIFT operation for target vehicle

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

flag_0 = True
trackedCars = []
time = 0
frameNum = 0
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) # Creteria for tracking 

while True:
    frameNum = frameNum + 1
    numStr = "The number of frame is:" + str(frameNum) + "."
    print numStr
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray,gray)

    time = time + 1
    if time > 5:    # Windows update time
        time = 0
        flag_0 = True
    
    cars = car_cascade.detectMultiScale(gray, scaleFactor=2.3, minNeighbors=20, minSize=(20,20), maxSize=(200,200))
    
    realCars = []
    roiCars = []
    overallCars = []

    for (x,y,w,h) in cars:

        flag_1 = False
        if isNotInCorrectPostion((x,y,w,h)):
            continue
        for (x_temp, y_temp, w_temp, h_temp) in cars:
            if isFullyInclude((x,y,w,h),(x_temp, y_temp, w_temp, h_temp)):
                flag_1 = True
                break
        if flag_1:
            continue
        else:
            realCars.append((x,y,w,h))
            roiImage = gray[y:y+h, x:x+w]
            roiImageTrack = img[y:y+h, x:x+w]
            roiInfo = [roiImage, (x,y,w,h)]
            roiCars.append(roiInfo)

            hsv_roi =  cv2.cvtColor(roiImageTrack, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            integrationCar = [roi_hist, (x,y,w,h)]
            overallCars.append(integrationCar)
    
    if not flag_0:
        for (roi_hist, (x,y,w,h)) in trackedCars:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            ret1, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)
            roiImage = gray[track_window[1]:track_window[1]+track_window[3], track_window[0]:track_window[0]+track_window[2]]
            roiImageTrack = img[track_window[1]:track_window[1]+track_window[3], track_window[0]:track_window[0]+track_window[2]]
            roiInfo = [roiImage, (x,y,w,h)]
            roiCars.append(roiInfo)
            realCars.append(track_window)

            hsv_roi =  cv2.cvtColor(roiImageTrack, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            integrationCar = [roi_hist, (x,y,w,h)]
            overallCars.append(integrationCar)
    
    trackedCars = overallCars

    if flag_0:
        flag_0 = False

    matchFlag = False
        
    for (roiImage, (x,y,w,h)) in roiCars:    # Vehicle matching
        kp, des = sift.detectAndCompute(roiImage, None)
        if (type(des) == type(None)):
            continue
        matches = bf.knnMatch(ObjDes,des, k=2)    # kNN algorithm used for matching
        if not (len(matches[0]) == 2):
            continue
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        num = len(good)
        if(num >= 5):    # Symbol for successfully matched
            if not inCorrectRegion(x,y):
                continue
            cv2.imshow('Object', roiImage)
            cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(0,255,255),2)
            matchFlag = True
            print "Successfully Matched!"
            break
                 
    for (x_real,y_real,w_real,h_real) in realCars:  # All real cars marked into image
        cv2.rectangle(img,(x_real,y_real),(x_real+w_real,y_real+h_real),(0,255,0),2)
    
    cv2.imshow('video', img)
    if not matchFlag:
        print "Matching Failed!"

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
