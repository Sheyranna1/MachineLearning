# importing the module
import cv2 
import numpy as np
import csv



# open the file in the write mode
f = open('coords.csv', 'w') # will throw error sometimes
fieldnames = [ 'Index', 'X', 'Y' ]
f.write(','.join(fieldnames) + '\n')

#This is where you enter the video you want the code to read
source = cv2.VideoCapture('MakoU.mp4')

#Object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=500)
# running the loop

index = 0
while True:
  
    index += 1
    # extracting the frames from the video//runs smoother?
    ret, img = source.read()
      
    #resizing super large video
    ret, frame = source.read() 
    down_width = 1200
    down_height = 700
    down_points = (down_width, down_height)

    ret, frame = source.read()  
    frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)
    
    lower_limit = np.array([190])
    upper_limit = np.array([255])
    
    height, width, _ = frame.shape 
    print(height,width)
    roi = frame[50:1300, 200:950]
    
    # run roi through 
    mask = object_detector.apply(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 250:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y),(x + w, y + h), (0, 255, 0), 2)
            
                          
    detections.append({x, y, w, h})
    
    f.write(f"{index},{x},{y}\n")
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
