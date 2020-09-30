from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from Recognition import ImageEnhancement
import threading
from multiprocessing import Queue
import numpy as np
import datetime
from attendance import AttendanceMarker
 
FRAME_WINDOW = 'Frame'
RECOGNITION_WINDOW = 'Recognition'
TEXT_WINDOW = 'Help'
BG_THREAD_NAME = 'bg_thread'
IMAGE_SUPER_RESOLUTION_METHOD = None

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

que =Queue()

attendance_marker = AttendanceMarker()

cv2.namedWindow(TEXT_WINDOW)
cv2.moveWindow(TEXT_WINDOW,320,700)
cv2.namedWindow(FRAME_WINDOW)
cv2.moveWindow(FRAME_WINDOW,320,135)
cv2.namedWindow(RECOGNITION_WINDOW)
cv2.moveWindow(RECOGNITION_WINDOW, 1000, 160) 

enhancement = ImageEnhancement(method=IMAGE_SUPER_RESOLUTION_METHOD)

image_to_process = None
processed_frame = np.zeros((224,224,3))
recognized_faces = []

def show_text_window(log = None):
	frame = np.zeros((250,1000,3),dtype=np.uint8)
	text = "Press r to recognise faces \nPress s to save the attendance \nPress q to quit"
	y0, dy = 20, 20
	for i, line in enumerate(text.split('\n')):
		y = y0 + i*dy
		cv2.putText(frame, line.strip(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (137, 243, 111), 1)
	if log:
		log = '[ logs ]\n \n' + log
		y0 = y + dy*3
		for i, line in enumerate(log.split('\n')):
			y = y0 + i*dy
			cv2.putText(frame, line.strip(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (101, 243, 224), 1)
	cv2.imshow(TEXT_WINDOW,frame)
show_text_window()


def log(text):
	print('[INFO] ' + text)
	show_text_window(log=text)

def post_process_frame(frame):
	frame = enhancement.improve_quality(frame)
	boxes = face_recognition.face_locations(frame,
		model=args["detection_method"])

	encodings = face_recognition.face_encodings(frame, boxes)

	names = []
	
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
		names.append(name)

	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	print('job completed!')
	return frame , names

bg_thread = None


while True:
	frame = vs.read()
	original_frame = frame.copy()
	image_to_process = original_frame
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=480)
	r = frame.shape[1] / float(rgb.shape[1])
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	for (top, right, bottom, left) in boxes:
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, 'face', (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

	if bg_thread and  not bg_thread.isAlive():
		if que:
			processed_frame, recognized_faces =que.get()
		bg_thread = None
	
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)

	if args["display"] > 0:
		cv2.imshow(FRAME_WINDOW, frame)
		cv2.imshow(RECOGNITION_WINDOW,processed_frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

		elif key == ord("r"):
			if bg_thread == None :
				bg_thread = threading.Thread(target=lambda q, arg1: q.put(post_process_frame(arg1)),name=BG_THREAD_NAME, args=(que, rgb))
				bg_thread.start()
				log('Recognising faces...')
			else:
				log('Recognition process already active..please wait.')

		elif key == ord('s'):
			attendance_marker.mark_attendance(recognized_faces)
			log('Marking attendance for {}'.format(recognized_faces))


cv2.destroyAllWindows()
vs.stop()


if writer is not None:
	writer.release()