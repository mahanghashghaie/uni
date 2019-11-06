import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1
    if ch == ord('2'):
        mode = 2
    if ch == ord('3'):
        mode = 3
    if ch == ord('4'):
        mode = 4
    if ch == ord('5'):
        mode = 5
    if ch == ord('6'):
        mode = 6
    if ch == ord('7'):
        mode = 7
    if ch == ord('q'):
        break

    if mode == 1:
        # just example code
        # your code should implement
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if mode == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    if mode == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    if mode == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if mode == 5:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        frame = th    
    if mode == 6:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    if mode == 7:
        frame = cv2.Canny(frame,50,100)
    # Display the resulting frame
    cv2.imshow('frame', frame)






# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
