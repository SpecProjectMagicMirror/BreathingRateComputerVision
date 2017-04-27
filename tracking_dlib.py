from imutils.video import VideoStream
import imutils
import cv2
import numpy as np
import os
import time
import dlib
from skimage import io

def face_landmarks(img):
    predictor_path = "../dlib/python_examples/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    dets = detector(img)

    #output face landmark points inside rectangle
    #shape is points datatype
    #http://dlib.net/python/#dlib.point
    for k, d in enumerate(dets):
        shape = predictor(img, d)

    vec = np.empty([68, 2], dtype = int)
    for b in range(68):
        vec[b][0] = shape.part(b).x
        vec[b][1] = shape.part(b).y

    return vec

#nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
#if nose_cascade.empty():
#    raise IOError('Unable to load the nose cascade classifier xml file')
    
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
    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)

    vec = face_landmarks(frame)
    nose_coord = vec[30:36]

    max_x = int(np.floor(1.035 * max(nose_coord[:,0])))
    max_y = int(np.floor(1.04 * max(nose_coord[:,1])))
    min_x = int(np.ceil(0.96 * min(nose_coord[:,0])))
    min_y = int(np.ceil(0.96 * min(nose_coord[:,1])))

    x,y,w,h = min_x, min_y, max_x-min_x, max_y-min_y
    bbox = (int(x),int(y),int(w),int(h))
    img = []

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
