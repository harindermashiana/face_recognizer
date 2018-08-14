# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:35:21 2018

@author: Harinder
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:00:18 2018

@author: Harinder
"""


import cv2
import matplotlib.pyplot as plt
import time
import os 
import numpy as np
 
#this function detects faces in an image and returns those faces along with the their co-ordinates
def detect_face2(img):
 #images are in BGR format and we need to conver image to gray scale because all the operations that we are going to do
 #they are only supported in gray scale format opencv
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 #We will use lbpcascade as it is the fastest one to detect images
 #speed will help in live face recognition
 face_cascade = cv2.CascadeClassifier(r'lbpcascade_frontalface_improved.xml')
 
 #the detected faces are retured as the bounding box coordinates of the faces in the image
 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);
 
 grays=[]
 face_cood=[]
 if (len(faces) == 0):
    return None, None

 else:
     for face in faces:
         x,y,w,h=face
         grays.append(gray[y:y+w, x:x+h])
         face_cood.append(face)
 return grays, face_cood


def prepare_training_data(data_folder_path):
 
 dirs = os.listdir(data_folder_path)
 

 faces = []
 labels = []
 #we will go through all the directories and images one by one
 for dir_name in dirs:
 
   if not dir_name.startswith("s"):
        continue;
   #removing the s from the folder name we get the label we want
   label = int(dir_name.replace("s", ""))
 
   dir_path = data_folder_path + "/" + dir_name
 
   images_names = os.listdir(dir_path)

   for image_name in images_names:
 
 
         image_path = dir_path + "/" + image_name
 
         image = cv2.imread(image_path)
 
         cv2.imshow("Training on image...", image)
         cv2.waitKey(100)
         #each image in the folder is read and fed to the face detection function
         face, rect = detect_face2(image)
         #for each face detected we add the bounding box coordinates of that face along with 
         #the corresponding label in two lists
         if face is not None:
            face=face[0]
            rect=rect[0]
            faces.append(face)
            labels.append(label)
 
         cv2.destroyAllWindows()
         cv2.waitKey(1)
         cv2.destroyAllWindows()
 
 return faces, labels




print("Preparing data...")
faces, labels = prepare_training_data(r"Enter here the path to the folder containg subdirectories of images")
print("Total number of  faces: ", len(faces))
print("Total number of labels: ", len(labels))

#we use LBP face recognizer and we feed the faces along with corresponding labels to it for training
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
labels=np.array(labels)   
face_recognizer.train(faces,labels)

#this funtion draws the bounding box around the detected face in the image
def draw_rectangle(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#this function is used to display the name over the bounding box
def draw_text(img, text, x, y):
 cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0),2)



#the name of the subjects that is people can be placed here according to folders
subject=["","harry","ankit","abhi"]

def predict(test_img):
    
 img = test_img.copy()
 faces, rects = detect_face2(img)
 if faces is None:
     return test_img
 else:
     #we use the for loop to loop over multiple faces detected in an image
     for face, rect in zip(faces,rects):
         
         label= face_recognizer.predict(face)
         label_text = subject[int(label[0])]
 
         draw_rectangle(img, rect)
         draw_text(img, label_text, rect[0], rect[1]-5)
 
 return img
#this function uses your webcam and takes frames from it and feeds those frames one by one
#to the face recognizer
def startcam():
      capture = cv2.VideoCapture(0)
      
      while True:
              stime = time.time()
              ret, frame = capture.read(0)
              
              frame = cv2.flip(frame, 1)
              if ret:
                      predicted_img1 = predict(frame)
                      cv2.imshow('frame', predicted_img1)
                      
                      print('FPS {:.1f}'.format(1 / (time.time() - stime)))
              if cv2.waitKey(50) & 0xFF == ord('q'):
                  break
      capture.release()
      cv2.destroyAllWindows()

startcam()
