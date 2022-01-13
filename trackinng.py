# importing the module
import cv2 
import numpy as np
import csv

fieldnames = [ 'X', 'Y' ]

rows = []

# open the file in the write mode
f = open('Test', 'w') # will throw error sometimes

#This is where you enter the video you want the code to read
source = cv2.VideoCapture('MakoUvideo (1).mp4')

#Object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=500)
# running the loop


while True:
  
    # extracting the frames from the video// makes the "video" run smoother?
    ret, img = source.read()
      
    # Code to convert your video to gray-scale
    # grayMovie = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # displaying the video
    # cv2.imshow("Live", gray)
    ret, frame = source.read()
    down_width = 1300
    down_height = 600
    down_points = (down_width, down_height)
    scale_up_x = 1.2
    scale_up_y = 1.2

    ret, frame = source.read()  
    frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)
    
    lower_limit = np.array([190])
    upper_limit = np.array([255])
    
    height, width, _ = frame.shape 
    print(height,width)
    roi = frame[0:1900, 1:1900]
    
    # masking
    mask = object_detector.apply(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y),(x + w, y + h), (0, 255, 0), 2)
            
                          
    detections.append({x, y, w, h})
    f.write(f"{x},{y}\n")
    print(detections)
    # exiting the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    cv2.imshow("roi", roi)
    #cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)  
# closing the window
cv2.destroyAllWindows()
source.release()
f.close()
