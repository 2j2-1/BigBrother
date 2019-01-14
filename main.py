import numpy as np
import cv2 as cv
import os
import datetime
# face_cascade = cv.CascadeClassifier('lbpcascade_frontalface_improved.xml')
face_recognizer = cv.createLBPHFaceRecognizer()
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
recognitionThreshold = 80
generateTraingDate = False
showOutput = True

def detectFace(classifier, img, scaleFactor=1.3):
	rect = classifier.detectMultiScale(img, scaleFactor, 5)
	return rect

def drawFaces(img,faces):
	for (x,y,w,h) in faces:
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	cv.imshow('frame', img)


def getFaces(img,faces):
	roi_color = []
	for (x,y,w,h) in faces:
		roi_color.append(img[y:y+h, x:x+w])
	return roi_color

def writeFaces(img,faces,folder):
	faces = getFaces(img,faces)
	for i in range(len(faces)):
		cv.imwrite('%s.jpg'%(folder) ,faces[i])


def CreateTrainingDataFaces(img,faces,folder):
	faces = getFaces(img,faces)
	labels = []
	for i in range(len(faces)):
		cv.imwrite('TrainingData/%s/%s.jpg'%(folder,str(datetime.datetime.now())) ,faces[i])
		labels.append(int(generateTraingDate.replace("n0","")))
	print "Adding %d new faces to model as %s" %(len(faces),generateTraingDate)
	face_recognizer.update(faces,np.array(labels))
	face_recognizer.save("faceData.yml")

def getTraingingData(folderPath):
	dirs = os.listdir(folderPath)
	faces = []
	labels = []
	for dirName in dirs:
		label = int(dirName.replace("n",""))
		subjectDirPath = folderPath + "/" + dirName
		subjectImageName = os.listdir(subjectDirPath)
		for imageName in subjectImageName:
			imagePath = subjectDirPath + "/" + imageName
			face = cv.imread(imagePath)
			face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
			faces.append(face)
			labels.append(label)
	return faces, labels

def predict(img,face,rect):
	imgCopy = img.copy()
	label = face_recognizer.predict(face)	

	if label[1] < recognitionThreshold:
		label = "N0" + str(label[0])
	else:
		label = "Unrecognised"

	if showOutput:
		draw_rectangle(imgCopy, rect)
		draw_text(imgCopy, label, rect[0], rect[1]-5)
	else:
		print label
	if generateTraingDate and label.lower() != generateTraingDate:
		CreateTrainingDataFaces(gray,[rect],generateTraingDate)
	elif label == "Unrecognised":
		writeFaces(gray,[rect],"Unrecongized/%s"%str(datetime.datetime.now()))

	return imgCopy

	

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 

def draw_text(img, text, x, y):
	cv.putText(img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)

def train():
	print "Preparing Data..."
	faces, labels = getTraingingData("TrainingData")
	print "Completed Preparation"
	print "Total faces:", len(faces)
	face_recognizer.train(faces,np.array(labels))
	face_recognizer.save('faceData.yml')
	print "Training Completed"

try:
	face_recognizer.load('faceData.yml')
except:
	train()
cap = cv.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = detectFace(face_cascade,gray)
	facesImg = getFaces(gray,faces)

	for face in range(len(facesImg)):
		frame = predict(frame,facesImg[face],faces[face])

	if showOutput:
		drawFaces(frame,faces)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()

