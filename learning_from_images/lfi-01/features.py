import cv2

mode = 0
cap = cv2.VideoCapture(0)
cv2.namedWindow('Learning from images: SIFT feature visualization')
while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key

    # YOUR CODE HERE
    
    ret, frame = cap.read()
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(frame,None)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('2'):
        mode = 2
    if ch == ord('3'):
        mode = 3
    if ch == ord('4'):
        mode = 4
    if ch == ord('q'):
        break
        
        
    if mode == 2:
        frame = cv2.drawKeypoints(gray_frame,kp,frame)
    if mode == 3:
        frame = cv2.drawKeypoints(gray_frame,kp,frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    if mode == 4:
        frame = cv2.drawKeypoints(gray_frame,kp,frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame', frame)
    
cap.release()
cv2.destroyAllWindows()