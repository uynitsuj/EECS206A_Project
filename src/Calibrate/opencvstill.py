# program to capture single image from webcam in python
# importing OpenCV library
import matplotlib.pyplot as plt
import time
import cv2
import datetime

cam = cv2.VideoCapture(0)

cv2.namedWindow("Video")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        now = datetime.datetime.now()
        filename = "image%d" % now.year+"-%d" % now.month+"-%d" % now.day + \
            "-%d" % now.hour+"-%d" % now.minute+"-%d" % now.second + ".jpg"
        # SPACE pressed
        cv2.imwrite(filename, frame)
        print("{} written!".format(filename))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
