from imutils.video import VideoStream
import imutils
import cv2
import numpy as np
import os
import time
from skimage import io

nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')
    
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
     
    tracker = cv2.Tracker_create("KCF")
 
    # Capture real-time video
    video = VideoStream(usePiCamera=False, resolution=(640, 480)).start()
    
    time.sleep(1)
    
    # Read first frame.
    frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    
    x,y,w,h = list(nose_rects[0])
    bbox = (int(x),int(y),int(w),int(h))

    time.sleep(1)
    start_time = time.time()
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    count = 0
    while True:
        # Read a new frame
        frame = video.read()

        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))
 
        img = frame[int(bbox[1]):int(bbox[1]+bbox[3]+1), int(bbox[0]):int(bbox[0]+bbox[2]+1)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray,(5,5),0)

        name = "/Users/rangodrich/CT_specialization_project/video_frames/frame%d.jpg"%count
        # Display result
        cv2.imshow("Tracking", frame)
        cv2.imwrite(name, blur_img)
        
        count += 1
        endtime = time.time() - start_time
        
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or endtime > 30: break
            
video.stop()
cv2.destroyAllWindows()
