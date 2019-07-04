# Importing the suitable libraries
import cv2
import face_recognition
import argparse
import time
import pickle
from imutils.video import VideoStream

# construct an argument parser
#ap = argparse.ArgumentParser()
#ap.add_argument("-e", "--encoding_path", required= True,help="Path to the location of the encoding")
#ap.add_argument("-o"."--output",type=str,help= "path to the output videos")
#ap.add_argument("-y","--display",type=int,default=1,help="Whether or not to display the output")
#ap.add_argument("-d","--detection_method",type= str,default ="cnn",help="model to use : hog or cnn")

args = vars(ap.parse_args())

print("[INFO]!! Loading the encodings...")
data = pickle.load(open(args["encoding_path"],"rb").read())
print("[INFO[!! Starting the video stream..")
vs= VideoStream(src=0).start()
writer= None
time.sleep(2.0)

while True:
	frame = vs.read()
	rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(rgb,width=750)
	r =frame.shape[1]/float(rgb.shape[1])
	boxes= face_recognition.face_locations(rgb,model ="cnn")#args["detection_method"])
	encodings =face_recognition.face_encodings(rgb,boxes)
	names=[]
	for encoding in encodings:
		matches = face_recognition.compare_faces("encoding.pickle",encoding)#data["encodings"]),encoding)
		name="Unknown"
		print(matches)
		if True in matches:
			matchesIdx =[i for (i,b) in matches if b]
			counts={}
			for i in matchesIdx:
				name = data["names"][i]
				counts[name]= counts.get(name,0)+1
			name =max(counts,key= count.get)
		names.append(name)
	for((top,right,bottom,left),name) in zip(boxes,names):
		top=int(top*r)
		right=int(right*r)
		bottom=int(bottom*r)
		left=int(left*r)
 		cv2.rectangle(frame,(left, top),(right, bottom),(0,255,0),2)
		y=top-15 if top-15>15 else top+15
		cv2.putText(frame,name ,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)	
cv2.destroyAllWindows()
vs.stop()

