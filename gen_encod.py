# Libraries
import numpy as np
import face_recognition
import argparse
import pickle
import cv2
import os

# Taking input from the command prompt
ap= argparse.ArgumentParser()
ap.add_argument("-i","--image_dir",default = os.path.join(os.getcwd(),"data"),help="Path to the image folder")
ap.add_argument("-e","--encoding" ,default = os.path.join(os.getcwd(),"encoding.pickle"), help="path to the encoding file")
ap.add_argument("-d","--det-method",default = "cnn",help ="Type of model to use")
args = vars(ap.parse_args())
print("[INFO] Got the input..")
image_path = args["image_dir"]
known_encodings = []
known_names = []
# Looping over the images and generating names and encoding
for i,image_p in enumerate(os.listdir(image_path)):
	image =cv2.imread(os.path.join(os.getcwd(),"data",image_p))
	name = image_p.split(".")[0]
	print(name)
	print(image_p)
	print("Processing images...")
	rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
	boxes = face_recognition.face_locations(rgb,model="cnn")
	encodings = face_recognition.face_encodings(rgb,boxes)
	for encoding in encodings:
		# Processsing the encodings
		known_encodings.append(encoding)
		known_names.append(name)
print("[INFO]!! Saving the encoding and name details ")
data = {"encoding": known_encodings, "names":known_names}
f =open(args["encoding"],"wb")
f.write(pickle.dumps(data))
f.close()

	
