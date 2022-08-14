import mtcnn
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input
import time

MTCNN = mtcnn.MTCNN(min_face_size = 15)
mask_model = keras.models.load_model('mask_detector.h5')

def detect_and_predict(frame):
    faces = MTCNN.detect_faces(frame)
    for face in faces:
        x,y,w,h = face['box']
        x2, y2 = x+w, y+h
        
        f_img=frame[y:y2,x:x2]
        f_img = cv2.resize(f_img,(224,224))
        f_img = preprocess_input(f_img)
        f_img = np.resize(f_img,(1,224,224,3))
        [[mask,no_mask]] = mask_model.predict(f_img)
        
        if mask>no_mask:
            cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),1)
            cv2.putText(frame,'MASK',(x,y2+20),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),1)
        else:
            cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),1)
            cv2.putText(frame,'NO MASK',(x,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
            
            
vid = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

while(True):
    ret, frame = vid.read()
        
    detect_and_predict(frame)
   
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    fps = int(fps)
    fps = str(fps)
    
    cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
