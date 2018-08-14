# face_recognizer
Opencv Face recognition
Prerequistes
1. opencv
2. matplotlib
3. numpy 

In order to perform face recognition, I have divided the task into following steps:

1. Preparing data for training our algorithm to recognize faces
2. Detect faces in the image
3. Feed those faces to our algorithm for training
4. Recognizing faces in the live video

Step 1 : Preparing of Data
You need to have images of different people stored under different directories and all these sub directories should be under one directory.
Images should be labelled 1 to n and should have either .png or .jpg format. Directory sstructure should look like this:

data
 |
 |
 --->s1
      |
      |
      ---->1.jpg
           2.jpg
           .
           .
           n.jpg
 --->s2
      |
      |
      ---->1.jpg
           2.jpg
           .
           .
           n.jpg

Step 2 : Detect faces in the image
I have use Linear Binary Patterns cascade for detecting the faces in the image due its better speed which will help with live face detection

Step 3: Train the face recognizer
I have used Linear Binary Patterns Histogram for recognizing the detected faces. It created Histograms of the faces it was trained and when 
given new faces it compares the Histogram it created of that image with the stored histograms for producing the results.

Step 4: Predicting faces in the live video
In order to recognize faces in the live video, we use openCv to get frames from the webcam and feed these frames to first detect faces and then
feed those detected faces to the LBPH face recognizer to recognize the faces.
