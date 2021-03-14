import cv2
import os
import imutils
algo = "haarcascade_frontalface_default.xml"
#load the algo file
haar = cv2.CascadeClassifier(algo)

#initialize video camera
cam = cv2.VideoCapture(0)
address = "https://192.168.2.107:8080/video"
cam.open(address)

while True: 
    _,frame = cam.read()
    frame = imutils.resize(frame, width = 500)
    #converting color to grayscale image
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Obtaining Face coordinates by passing algorithm
    faces = haar.detectMultiScale(grayFrame, 1.3,4)
    
    for (x,y,w,h) in faces:
        #draw rectangle on the face
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)   
        #display
    cv2.imshow("face-detection", frame)
    key = cv2.waitKey(1)
    #27 = esc btn
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
    
    
