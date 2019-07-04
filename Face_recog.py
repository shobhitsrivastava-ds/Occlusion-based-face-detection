# Importing libraries
import cv2
import numpy as np
import pickle 
import os
import argparse
import face_recognition

# Argument Parser
ap =argparse.ArgumentParser()
ap.add_argument("-e", "--encoding",default =os.path.join(os.getcwd(),"encoding.pickle"), help= "path to the encoding")
ap.add_argument("-i", "--image",default=os.path.join(os.getcwd(),"index.png"), help= "path to the input image")
ap.add_argument("-d", "--detection-method", default="cnn", help="type of model")

# Getting the data
args = vars(ap.parse_args())
print("[INFO]!! Testing the image !!")
data = pickle.loads(open(args["encoding"],"rb").read())
image = cv2.imread(os.path.join(os.getcwd(),"mk.jpg"))
rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
boxes= face_recognition.face_locations(rgb,model= "cnn")
encodings = face_recognition.face_encodings(rgb,boxes)
names=[]

for encoding in encodings:
	matches = face_recognition.compare_faces(data["encoding"],encoding)
	name = "Unknown"
	if True in matches:
		match_idx = [i for (i,b) in enumerate(matches) if b]
		counts= {}
		for i in match_idx:
			name = data["names"][i]
			counts[name] =counts.get(name,0)+1
		name = max(counts, key= counts.get)
	names.append(name)
for ((top,right,bottom,left),name) in zip(boxes,names):
	cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
	y = top-15 if top-15>15 else top+15
	cv2.putText(image,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
	
