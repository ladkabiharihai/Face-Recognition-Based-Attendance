import face_recognition
import argparse
import imutils
import pickle
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())



print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())


print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None

i = 0

while True:
	(grabbed, frame) = stream.read()
	if not grabbed:
		break

	if i % 300 != 0:
		i += 1
		continue
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	for (top, right, bottom, left) in boxes:
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 24,
			(frame.shape[1], frame.shape[0]), True)
	if writer is not None:
		writer.write(frame)
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
stream.release()
if writer is not None:
	writer.release()