import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

# tello.takeoff()
cv2.imwrite("../data/image/handpose.png", frame_read.frame)

# tello.land()