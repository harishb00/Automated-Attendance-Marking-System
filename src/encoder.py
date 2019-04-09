import os
import pickle
import face_recognition as fr

#Access the directories which holds the db and Test Images

base_dir = 'assets/'
db_dir = os.path.join(base_dir, 'student_images')

image_names = [image_name for image_name in os.listdir(db_dir)]

encodings = {}

for image_name in image_names:
  #find the path of the image
  image_path = os.path.join(db_dir, image_name)
  
  #read the image
  image = fr.load_image_file(image_path)
  
  #find the facial encoding of the image
  encoding = fr.face_encodings(image)[0]
  
  #get the reg_no of the image
  #[:-4] changes 1054.jpg to 1054 . It removes last 4 characters '.jpg'. According to your extension length, modify it.
  reg_no = image_name[:-4] 
  
  #add the encoding to the dictionary with reg_no as the key
  encodings[reg_no] = encoding
  
fh = open('encodings.pickle', 'wb')
pickle.dump(encodings, fh)
fh.close()
