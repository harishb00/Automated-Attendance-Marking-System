import numpy as np
import face_recognition as fr
import pickle

#load the encodings file which returns a dictionary
fh = open('encodings.pickle', 'rb')
encodings = pickle.load(fh)
fh.close()

#extract keys (roll numbers)
reg_nos = list(encodings.keys())

#extract values (face encodings)
facial_encodings = list(encodings.values())

test_image_path = 'path_to_image.jpg'

#read the image from the specified path
test_image = fr.load_image_file(test_image_path)

#find face encodings for each face in the test image
test_encodings = fr.face_encodings(test_image)

records = []

#for each unknown face find the roll number
for test_encoding in test_encodings:
  distances = fr.face_distance(facial_encodings, test_encoding)
  pos = np.argmin(distances)
  reg_no = reg_nos[pos]
  records.append([reg_no, 'P'])

import csv
from datetime import datetime

#sets the file name to current date.csv
file_name = str(datetime.now().date()) + '.csv'

#create a new CSV file and write details about students who are present as '1054', 'P'
fh = open(file_name, 'w')
writer = csv.writer(fh, delimiter=',')
writer.writerow(['Reg No.', 'Status'])

#for each record, write the record as a row to the CSV file
for record in records:
  writer.writerow(record)

#close the file
fh.close()
