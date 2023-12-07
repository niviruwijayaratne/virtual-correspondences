import os
import cv2

cap = cv2.VideoCapture('two.mp4')

cnt = 0
while(cap.isOpened()):
    # fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if ret == True:
        resized = cv2.resize(frame, (640,360))
        cv2.imwrite(f"two_frames/{cnt}.jpg", resized)
        cnt += 1
    
    else:
        break

cap.release()