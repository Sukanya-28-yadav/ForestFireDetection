
import numpy as np
import cv2
'''
Fire_Reported = 0

img = cv2.imread("Dataset/Fire/resized_frame0.jpg")

blur = cv2.GaussianBlur(img, (21, 21), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
lower = [0, 0, 200]
upper = [50, 88, 226]
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")
mask = cv2.inRange(hsv, lower, upper)
output = cv2.bitwise_and(img, hsv, mask=mask)
no_red = cv2.countNonZero(mask)
'''
'''
print(no_red)
if int(no_red) > 15000:
    Fire_Reported = Fire_Reported + 1
if Fire_Reported >= 1:
    print("fire")
cv2.imshow('masj', mask)
cv2.imshow('output', output)
cv2.imshow('img', img)
cv2.waitKey(0)
'''
'''
contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,x+h),(0,0,255),5)
cv2.imshow('output', output)    
cv2.imshow('masj', mask)    
cv2.imshow("img ",img)
cv2.waitKey(0)
'''

import numpy as np
import cv2  

frame = cv2.imread('Dataset/Fire/resized_frame10004.jpg')
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of red color in HSV
lower_red = np.array([0, 10, 120])
upper_red = np.array([15, 255, 255])

mask = cv2.inRange (hsv, lower_red, upper_red)
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    for c in contours:
        #red_area = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w >= 5 and h >= 10:
            cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255), 2)


cv2.imshow('frame', frame)
cv2.imshow('mask', mask)

cv2.waitKey(0)
