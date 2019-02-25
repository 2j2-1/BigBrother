import cv2 as cv
import os
import datetime

cam1 = cv.VideoCapture(0)

while True:
	frame = cam1.read()[1]
	cv.imshow("Frame",frame)
	cv.imwrite('Monitor/%s.jpg'%(str(datetime.datetime.now())) ,frame)
	if cv.waitKey(1) & 0xFF == ord('q'):
			break