import numpy as np
import cv2 as cv
import os
import datetime
import json
import requests
url = "https://magnus.cjax.uk/reports/student/"
headers = {'Content-Type': 'application/json','Token': 'fd4917ce3f1e9bb8be8014f26b9fe17e7aaedb0b', 'Address': '20:16:B9:6C:E8:55'}


class Recognision():

	def __init__(self):
		self.trainingDataLocation = "TrainingData"
		self.face_recognizer = cv.face.LBPHFaceRecognizer_create()
		if os.path.exists('faceData.yml'):
			self.face_recognizer.read('faceData.yml')
		else:
			print "Failed to load face data"
			self.trainModel()
		
	def trainModel(self):
		print "Preparing Data..."

		faces, labels = self.getTraingingData(self.trainingDataLocation)

		print "Completed Preparation"
		print "Total faces:", len(faces)

		self.face_recognizer.train(faces,np.array(labels))
		self.face_recognizer.save('faceData.yml')

		print "Training Completed"

	def getTraingingData(self,folderPath):
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

	def CreateTrainingDataFaces(self,faceImg,userName):
		label = int(userName.replace("n0",""))

		self.writeFaces(self.trainingDataLocation,userName,faceImg)
		print "Adding new face to model as %s" %(userName)
		self.face_recognizer.update([faceImg],np.array([label]))
		self.face_recognizer.save("faceData.yml")

	def predict(self,face):
		return self.face_recognizer.predict(face)

	def writeFaces(self,folder,userName,face,img=None):
		if userName != "":
			folder = '%s/%s'%(folder,userName)
		existsOrCreate('%s'%(folder))
		cv.imwrite('%s/%s.jpg'%(folder,str(datetime.datetime.now())) ,face)


class Stream():

	def __init__(self, detector, recognisor):
		self.detector = detector
		self.recognisor = recognisor
		self.recognitionThreshold = 60
		self.user = ""
		

	def run(self,frame,user = ""):
		self.user = user
		self.frame = frame
		self.drawing = self.frame.copy()
		self.gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
		self.faces = self.detector.detectFace(self.gray)
		self.facesImg = self.detector.getFaces(self.gray, self.faces)

		for face in range(len(self.facesImg)):
			self.predict(self.facesImg[face], self.faces[face])

	def predict(self,faceImg,faceRect):
		label, value = self.recognisor.predict(faceImg)
		if self.user == "":
			self.recognitionThreshold = 80
		else:
			self.recognitionThreshold = 60

		if value < self.recognitionThreshold:
			label = "N0" + str(label)
		else:
			label = "Unrecognised"
			value = 100
		if label!="Unrecognised" and self.user == "":
			cv.imwrite('Monitored/%s-%s.jpg'%(label,str(datetime.datetime.now())) ,faceImg)
			data = {"camera_id":"1","identifier":label,"notes":value,}
			r = requests.post(url+"success/", data=json.dumps(data), headers=headers)
			print r

		elif self.user != "" and self.user != label.lower():
			self.recognisor.CreateTrainingDataFaces(faceImg,self.user)
		else:
			self.recognisor.writeFaces("Unrecongized","",faceImg)
			data = {"camera_id":"1","identifier":label,"notes":value,}
			r = requests.post(url+"unsuccessful/", data=json.dumps(data), headers=headers)
			print r

	def draw_face(self,rect,text,value):
		x, y, w, h = rect
		cv.rectangle(self.drawing, (x, y), (x+w, y+h), (255, 0, 0), 2)
		cv.putText(self.drawing, text+": "+str(int(100-(value)))+"%", (x, y-5), cv.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)

class Detection():

	def __init__(self):
		self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

	def detectFace(self,img, scaleFactor=1.3):
		rect = self.face_cascade.detectMultiScale(img, scaleFactor, 5)
		return rect

	def getFaces(self,img,rects):
		roi_color = []
		for (x,y,w,h) in rects:
			roi_color.append(img[y:y+h, x:x+w])
		return roi_color

def existsOrCreate(fileLocation):
	if not os.path.isdir(fileLocation):
		os.makedirs(fileLocation)

detect = Detection()
recognision = Recognision()
stream = Stream(detect,recognision)

folders = ["Monitor","toAdd"]
while True:
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
	for i in folders:
		dirs = os.listdir(i)
		if len(dirs)>5:
			for imageName in dirs:
				imagePath = i + "/" + imageName
				face = cv.imread(imagePath)
				try:
					if i == "toAdd":
						stream.run(face,imageName[:8])
					else:
						stream.run(face)
					os.remove(imagePath)
				except:
					break
				if cv.waitKey(1) & 0xFF == ord('q'):
						break
cv.destroyAllWindows()

