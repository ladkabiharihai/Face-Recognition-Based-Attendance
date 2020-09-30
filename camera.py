import time
import cv2
import imutils

from imutils.video import VideoStream

vs = VideoStream(src='http://192.168.43.1:8080/video').start()
frame_num = 0
avg_fps = 0

while True:
    start_time = time.time()
    frame = vs.read()
    frame = imutils.resize(frame,width=1080)
    cv2.imshow('frame',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    end_time = time.time()
    current_fps = 1 / (end_time - start_time)
    avg_fps = (avg_fps * frame_num + current_fps) / (frame_num + 1)
    frame_num += 1

    print('FPS = {:.2f}'.format(avg_fps))

cv2.destroyAllWindows()
vs.stop()