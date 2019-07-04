# Importing libraries
import cv2
import numpy as np
import pickle 
import os
import argparse
import face_recognition
import time
# Argument Parser
ap =argparse.ArgumentParser()
ap.add_argument("-e", "--encoding",default =os.path.join(os.getcwd(),"encod.pickle"), help= "path to the encoding")
ap.add_argument("-i", "--image",default=os.path.join(os.getcwd(),"index.png"), help= "path to the input image")
ap.add_argument("-d", "--detection-method", default="hog", help="type of model")

# Getting the data
args = vars(ap.parse_args())
print("[INFO]!! Testing the image !!")
data = pickle.loads(open(args["encoding"],"rb").read())

print("Starting the vidoes")


vid = cv2.VideoCapture(0)#os.path.join(os.getcwd(),"Suits.mkv"))
#time.sleep(2.0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output4.mkv',fourcc, 20.0,(340,280))
#out = cv2.VideoWriter('out.avi',fourcc, 10.0, (540,380))
while True:
	#image = cv2.imread(os.path.join(os.getcwd(),"index_5.jpg"))
	_,image = vid.read()
	#image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
	image = cv2.resize(image,(110,90))
	rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	boxes= face_recognition.face_locations(rgb,model= "hog")
	encodings = face_recognition.face_encodings(rgb,boxes)
	names=[]
	print("looking for encodings")
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
		y = top if top-15>15 else top+15
		cv2.putText(image,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.25,(0,255,0),1)
	image = cv2.resize(image,(340,280))
	cv2.imshow("Image",image)
	out.write(image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
vid.release()
cv2.destroyAllWindows()
	
