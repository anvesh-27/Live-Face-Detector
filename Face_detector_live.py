import cv2
from random import randrange

#Load pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#To capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:
    
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    #Drawing rectangles
    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

    #Displaying the frame
    cv2.imshow("Face Detector", frame)
    key = cv2.waitKey(1)                 # 1 means that opencv waits 1 milliseconds bfore giving a key press input to the program

    #Qutting live face detection , Press q to quit.
    if key==81 or key ==113:
        break

#Release the video capture object 
webcam.release()

print("Code completed")
