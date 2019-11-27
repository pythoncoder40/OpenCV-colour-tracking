import cv2
import numpy as np

def nothing (x) :
    pass

cv2.namedWindow('Tracking')

cv2.createTrackbar('Lower Hue', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower Sat', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Lower Val', 'Tracking', 0, 255, nothing)
cv2.createTrackbar('Upper Hue', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper Sat', 'Tracking', 255, 255, nothing)
cv2.createTrackbar('Upper Val', 'Tracking', 255, 255, nothing)

cap = cv2.VideoCapture(0)

while True :
    #frame = cv2.imread('bloblas.png')
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hue = cv2.getTrackbarPos('Lower Hue', 'Tracking')
    lower_sat = cv2.getTrackbarPos('Lower Sat', 'Tracking')
    lower_val = cv2.getTrackbarPos('Lower Val', 'Tracking')

    upper_hue = cv2.getTrackbarPos('Upper Hue', 'Tracking')
    upper_sat = cv2.getTrackbarPos('Upper Sat', 'Tracking')
    upper_val = cv2.getTrackbarPos('Upper Val', 'Tracking')

    lower_bound = np.array([lower_hue, lower_sat, lower_val])
    upper_bound = np.array([upper_hue, upper_sat, upper_val])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    key = cv2.waitKey(1)
    if key == 27 :
        break
cap.release()
cv2.destroyAllWindows()
