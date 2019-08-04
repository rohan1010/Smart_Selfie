from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2

def smile(mouth):
	A = dist.euclidean(mouth[3], mouth[9])
	B = dist.euclidean(mouth[2], mouth[10])
	C = dist.euclidean(mouth[4], mouth[8])
	avg = (A+B+C)/3
	D = dist.euclidean(mouth[0], mouth[6])
	mar=avg/D
	return mar

def eyes(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
 
	ear = (A + B) / (2.0 * C)
	return ear

COUNTER = 0
EYE_COUNTER = 0
TOTAL = 0

shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)


(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src='http://192.168.43.98:8080/video').start()
time.sleep(1.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		mouth= shape[mStart:mEnd]
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mar = smile(mouth)
		ear = (eyes(leftEye) + eyes(rightEye))/2
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		if (mar > .5 and mar < .8) or (mar> .30 and mar < .37) :
			COUNTER += 1
			if COUNTER >= 25:
				TOTAL += 1
				frame = vs.read()
				time.sleep(.3)
				img_name = "img/smile_{}.png".format(TOTAL)
				cv2.imwrite(img_name, frame)
				print("{} written!".format(img_name))
				COUNTER = 0
		if (ear > 0.1 and ear < 0.3):
			EYE_COUNTER += 1
			if EYE_COUNTER >= 5:
				TOTAL += 1
				frame = vs.read()
				time.sleep(.3)
				img_name = "img/blink_{}.png".format(TOTAL)
				cv2.imwrite(img_name, frame)
				print("{} written!".format(img_name))
				EYE_COUNTER = 0

	cv2.imshow("Frame", frame)

	if cv2.waitKey(1) == ord('q'):
		break

cv2.destroyAllWindows()
vs.stop()