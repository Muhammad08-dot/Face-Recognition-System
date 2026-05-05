import cv2
import os

# FIX 1: Ensure the dataset directory exists automatically
if not os.path.exists('dataset'):
    os.makedirs('dataset')

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

# FIX 2: Use an absolute path for the cascade if it's still not detecting
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(cascade_path)

face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0
while(True):
    ret, img = cam.read()
    
    # FIX 3: Change -1 to 1 (horizontal flip) or comment it out if your cam is normal
    img = cv2.flip(img, 1) 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image
        file_name = "dataset/User." + str(face_id) + '.' + str(count) + ".jpg"
        cv2.imwrite(file_name, gray[y:y+h,x:x+w])
        print(f"[DEBUG] Saving image {count} to {file_name}") # This confirms it's working

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 30: 
         break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()