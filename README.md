# Automated-Attendance-Marking-System
This repository contains the source code and explanation for 'Automated Attendance Marking System' project.

## Introduction
It’s always a hard task to take attendance in class rooms because of the strength of the class or due to proxies. It's just the repetitive task a faculty does all over the semester. If there's an application to capture the students and produce the attendance report automatically, it will be of great use. It reduces the time abruptly. Instead of calling out roll numbers of students manually, it's easy to capture the environment totally and then produce the attendance report based on that image. Convolutional Neural Networks solves many problems in Computer Vision. Already, it performs better in tasks like Object Detection, Object Recognition. This system uses Convolutional Neural Networks. In this project, the application takes class room picture as an input and produces the roll numbers as output. It will be a great tool for taking attendance by the faculties. It solves problems like more time consumption, proxies. The time can be rather utilised for teaching purposes. 

## Approach
### Encoding
```
Get access to db with individual images of students
for each student image in the db:
	find the face encoding of face
	add it to the dictionary as (roll no, face encoding)
store the dictionary in the pickle format
```

### Decoding
```
Get test image
load the pickle file which returns the dicionary (roll no, face encoding)
find face locations
for each face location:
	find the face encoding of the face
	find the distance between new face encoding and all known encodings in the dictionary
	find the encoding with minimum distance
	store the roll number of the minimum encoding
generate .csv file for the stored roll numbers
```

### Image Format
- db -> create a folder with all the student id images as in the id card/ student profile. Image name should be the roll number with the image extension.

### How to Use?
- First, run the 'encoder.py' file to find 'face encodings' of all the student profile images in 'db' directory. The output is stored as a pickle file in the specified path.
- Second, run the 'decoder.py' file to generate the attendace report as a CSV file. The pickle file which holds encodings created by previous step should also be given as input to the 'decoder.py' file.

### Libraries
- face_recognition
- numpy
