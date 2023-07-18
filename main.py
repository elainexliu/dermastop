import cv2

#load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#pre-trained hand cascade
hand_cascade = cv2.CascadeClassifier('cascades/hand.xml')

#opening webcam
video_capture = cv2.VideoCapture(0)

def hand_at_face(hands, faces):
    for (x, y, w, h) in hands:
        for (x1, y1, w1, h1) in faces:
            if x > x1 and y > y1 and (x+w) < (x1+w1) and (y+h) < (y1+h1):
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

while True:
    #frame by frame
    ret, frame = video_capture.read()
    #frame to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces and hands in frame
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    hands = hand_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #drawing rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #check if any detected hand is close to the face
    hand_at_face(hands, faces)    
    
    #display resulting frame
    cv2.imshow("face and Hand Detection", frame)

    #exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()