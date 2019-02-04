import numpy as np
import cv2 as cv
import os
import datetime
# face_cascade = cv.CascadeClassifier('lbpcascade_frontalface_improved.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
recognitionThreshold = 80
generateTraingData = False

TrainingDataFolder = "TrainingData"
TraningDataSaveLocation = "TrainingData/%s"%(generateTraingData)


showOutput = True

# Utility

def existsOrCreate(fileLocation):
	if not os.path.isdir(fileLocation):
		os.makedirs(fileLocation)

# Drawing Functions 

def draw_face(img, rect,text="",value=100):
	(x, y, w, h) = rect
	cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	if text != "":
		cv.putText(img, text+": "+str(100-value)+"%", (x, y-5), cv.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)

# Detection

def detectFace(classifier, img, scaleFactor=1.3):
	rect = classifier.detectMultiScale(img, scaleFactor, 5)
	return rect

def getFaces(img,rects):
	roi_color = []
	for (x,y,w,h) in rects:
		roi_color.append(img[y:y+h, x:x+w])
	return roi_color

# Io 

def writeFaces(folder,faces=[],img=None):
	existsOrCreate(folder)

	if faces == []:
		faces = getFaces(img,faces)

	for i in range(len(faces)):
		cv.imwrite('%s/%s.jpg'%(folder,str(datetime.datetime.now())) ,faces[i])


# Recognision

def CreateTrainingDataFaces(img,rects,folder):
	faces = getFaces(img,rects)
	label = int(generateTraingData.replace("n0",""))
	labels = [label] * len(faces)
	writeFaces(TraningDataSaveLocation,faces)
	print "Adding %d new faces to model as %s" %(len(faces),generateTraingData)
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
	label, value = face_recognizer.predict(face)

	if value < recognitionThreshold:
		label = "N0" + str(label)
	else:
		label = "Unrecognised"
		value = 100

	if showOutput:
		draw_face(imgCopy, rect, label, value)
	else:
		print label

	if generateTraingData and label.lower() != generateTraingData:
		CreateTrainingDataFaces(gray,[rect],generateTraingData)
	elif label == "Unrecognised":
		writeFaces("Unrecongized/",[rect],gray)

	return imgCopy

def train():
	print "Preparing Data..."
	faces, labels = getTraingingData(TrainingDataFolder)
	print "Completed Preparation"
	print "Total faces:", len(faces)

	face_recognizer.train(faces,np.array(labels))
	face_recognizer.save('faceData.yml')
	print "Training Completed"


# Start 
if os.path.exists('faceData.yml'):
	face_recognizer.read('faceData.yml')
else:
	print "Failed to load face data"
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
		cv.imshow('frame', frame)

	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()

