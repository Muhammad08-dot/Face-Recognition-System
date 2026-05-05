'''
Real Time Face Recognition - Security Version
    ==> Known Users (1-5): Green Box + Name
    ==> Unknown Users: Red Box + Auto-Save Photo to 'intruders/'
'''

import cv2
import numpy as np
import os 

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the LBPH Recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = os.path.join(current_dir, 'trainer', 'trainer.yml')

if not os.path.isfile(model_path):
    print("\n [ERROR] trainer.yml not found. Please run training (Step 2) first.")
    exit()

recognizer.read(model_path)

# Load the cascade classifier for face detection
cascadePath = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Names list: Index 1-5 correspond to User IDs 1-5
names = ['None', 'User 1', 'User 2', 'User 3', 'User 4', 'User 5'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Width
cam.set(4, 480) # Height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print("\n [INFO] Starting Recognition. Press 'ESC' to exit.")

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1) # Flip horizontally for natural mirror view
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    for(x, y, w, h) in faces:
        # Predict the ID and confidence (distance)
        id_num, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # LBPH Distance < 100 means the face is "Known"
        if (confidence < 100):
            if id_num < len(names):
                label = names[id_num]
            else:
                label = f"User {id_num}"
            
            box_color = (0, 255, 0) # Green for authorized users
            display_confidence = "  {0}%".format(round(100 - confidence))
        
        # If Distance > 100, the face is "Unknown"
        else:
            label = "UNKNOWN"
            box_color = (0, 0, 255) # Red for strangers
            display_confidence = "  {0}%".format(round(100 - confidence))
            
            # Create intruder folder and save photo
            if not os.path.exists('intruders'):
                os.makedirs('intruders')
            cv2.imwrite(os.path.join(current_dir, "intruders", "alert.jpg"), img)

        # Draw the rectangle and text on the screen
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(img, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(display_confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)  
    
    cv2.imshow('camera', img) 

    # Press 'ESC' for exiting video
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

# Cleanup
print("\n [INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()