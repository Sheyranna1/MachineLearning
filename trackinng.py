# importing the module
import cv2
import numpy as np
  
#This is where you enter the video you want the code to read
source = cv2.VideoCapture('mako.MOV')
#frames = np.array(source)
#gray = cv2.cvtColor(cv2.UMat(source), cv2.COLOR_RGB2GRAY)

#Object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=500)
# running the loop
while True:
  
# let's downscale the image using new  width and height
    

    # extracting the frames from the video// makes the "video" run smoother?
    ret, img = source.read()
    
    down_width = 1300
    down_height = 600
    down_points = (down_width, down_height)
    scale_up_x = 1.2
    scale_up_y = 1.2

    ret, frame = source.read()  
    frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    lower_limit = np.array([240])
    upper_limit = np.array([255])
    
    
    height, width, _ = frame.shape 
    #print(height,width)
    roi = frame[0:1900, 1:1900]
   
    
    # masking
    mask2 = cv2.inRange(gray, lower_limit, upper_limit)
    
    mask = object_detector.apply(mask2)#, cv2.inRange(gray, lower_limit, upper_limit)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y),(x + w, y + h), (100, 155, 0), 2)

    detections.append({x, y})

    print(detections)

    # exiting the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    #cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)  
    #cv2.imshow("Color ", mask2)
# closing the window
cv2.destroyAllWindows()
source.release()