#### USING THE OPENCV CASCADE FILES

# import cv2

# #load the pre-trained face cascade
# face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# #pre-trained hand cascade
# hand_cascade = cv2.CascadeClassifier('cascades/hand.xml')

# #opening webcam
# video_capture = cv2.VideoCapture(0)

# def hand_at_face(hands, faces):
#     for (x, y, w, h) in hands:
#         for (x1, y1, w1, h1) in faces:
#             if x > x1 and y > y1 and (x+w) < (x1+w1) and (y+h) < (y1+h1):
#                 cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

# while True:
#     #frame by frame
#     ret, frame = video_capture.read()
#     #frame to greyscale
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     #detect faces and hands in frame
#     faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     hands = hand_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     #drawing rectangles around detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     #check if any detected hand is close to the face
#     hand_at_face(hands, faces)    
    
#     #display resulting frame
#     cv2.imshow("face and Hand Detection", frame)

#     #exit loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()



##### HAND PICKING DETECTION (INDEX IS CLOSE TO THUMB)

# import cv2
# import mediapipe as mp

# # Load the MediaPipe hand detection model
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# # Open the webcam
# video_capture = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     # Flip the frame horizontally for a mirror effect
#     frame = cv2.flip(frame, 1)

#     # Convert the frame to RGB for Mediapipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect hands in the frame
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Get the coordinates of the index and thumb fingertips
#             index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#             thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

#             # Check the distance between hand and lips
#             lip_distance = abs(thumb_fingertip.y - index_fingertip.y)

#             # Set a threshold distance to consider it as hand near the lips
#             threshold = 0.04

#             if lip_distance < threshold:
#                 # Draw bounding box around the hand
#                 box_size = 50
#                 x = int(index_fingertip.x * frame.shape[1])
#                 y = int(index_fingertip.y * frame.shape[0])
#                 cv2.rectangle(frame, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('Face and Hand Detector', frame)

#     # Exit loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the windows
# video_capture.release()
# cv2.destroyAllWindows()

##### face+lip based on face detection ###########################
# import cv2
# import dlib
# import mediapipe as mp

# # Load the Dlib face detector
# detector = dlib.get_frontal_face_detector()

# # Load the MediaPipe hand detection model
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# # Open the webcam
# video_capture = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     # Flip the frame horizontally for a mirror effect
#     frame = cv2.flip(frame, 1)

#     # Convert the frame to RGB for Mediapipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     for face in faces:
#         # Get the face bounding box coordinates
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()

#         # Get the coordinates of the lips
#         lip_top = (x+int(0.33 * w), y + int(0.65 * h))
#         lip_bottom = (x+int(0.66 * w), y + int(0.9 * h))

#         # Detect hands in the frame
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Get the coordinates of the index and thumb fingertips
#                 index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#                 print("index", index_fingertip)
#                 print("thumb", index_fingertip)
#                 print("lip bottom", lip_bottom)
#                 print("lip top", lip_top)

#                 # Check if the hand is near the lips
#                 if lip_bottom[1] < thumb_fingertip.y * frame.shape[0] < lip_top[1] and lip_bottom[1] < index_fingertip.y * frame.shape[0] < lip_top[1]:
#                     # Draw bounding box around the hand
#                     box_size = 50
#                     x = int(index_fingertip.x * frame.shape[1])
#                     y = int(index_fingertip.y * frame.shape[0])
#                     cv2.rectangle(frame, (x - box_size, y - box_size), (x + box_size, y + box_size), (0, 0, 255), 2)

#         # Draw bounding box around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Draw bounding box around the lips
#         cv2.rectangle(frame, lip_top, lip_bottom, (255, 0, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Face and Hand Detector', frame)

#     # Exit loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the windows
# video_capture.release()
# cv2.destroyAllWindows()


#################################### WORKS ##########################
import cv2
import dlib
import mediapipe as mp

# Load the Dlib face detector
detector = dlib.get_frontal_face_detector()

# Load the MediaPipe hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the webcam
video_capture = cv2.VideoCapture(0)

# Get the width and height of the webcam frame
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

def near(lip_top, lip_bottom, normalized_thumb, normalized_index):
    if lip_top[1] < normalized_index[1] < lip_bottom[1] and lip_top[0] < normalized_index[0] < lip_bottom[0]:
        if abs(normalized_thumb[1]/height - normalized_index[1]/height) < 0.04:
            return True
    return False

# if lip_top[1] < normalized_thumb[1] < lip_bottom[1] and lip_top[1] < normalized_index[1] < lip_bottom[1] and lip_top[0] < normalized_thumb[0] < lip_bottom[0] and lip_top[0] < normalized_index[0] < lip_bottom[0]:


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Get the face bounding box coordinates
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Normalize the lip coordinates
        lip_top = (x + int(0.25 * w), y + int(0.65 * h))
        lip_bottom = (x + int(0.75 * w), y + int(0.9 * h))

        # Detect hands in the frame
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the index and thumb fingertips
                index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Normalize the hand coordinates
                normalized_index = (index_fingertip.x*width, index_fingertip.y * height)
                normalized_thumb = (thumb_fingertip.x*width, thumb_fingertip.y * height)

                print("index", normalized_index)
                print("thumb", normalized_thumb)
                print("lip bottom", lip_bottom)
                print("lip top", lip_top)

                # Check if the hand is near the lips
                if near(lip_top, lip_bottom, normalized_thumb, normalized_index):
                    # Draw bounding box around the hand
                    box_size = 30
                    tempx = int(normalized_index[0]) #int(index_fingertip.x * width)
                    tempy = int(normalized_index[1]) #int(index_fingertip.y * height)
                    cv2.rectangle(frame, (tempx - box_size, tempy - box_size), (tempx + box_size, tempy + box_size), (0, 0, 255), 2)
                    print("DETECTED")

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw bounding box around the lips
        cv2.rectangle(frame, lip_top, lip_bottom, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face and Hand Detector', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
