import cv2
import numpy as np
import time
from threading import Thread
from djitellopy import Tello



class Frame_Receiver:
    def __init__(self, use_robot, Width, Height, record_video=False, video_name="video.avi"):
        self.use_robot = use_robot
        self.tello = None
        self.web_cam_cap = None
        self.frameWidth = Width
        self.frameHeight = Height
        self.image_to_video = np.zeros((self.frameHeight, self.frameWidth, 3))
        self.initialize_frame_cap()
        self.recorder = Thread(target=self.videoRecorder)
        self.record_video = record_video
        self.video_name = video_name
        if record_video:
            self.recorder.start()
        self.keep_recording = True

    def init_robot(self):
        print("Initialize Robot!")
        self.tello = Tello()
        self.tello.connect()
        print("battery level is: ", self.tello.get_battery())
        self.tello.streamon()
        # self.tello.set_video_resolution("low")
        # self.tello.set_video_fps("high")

    def init_web_cam(self):
        self.web_cam_cap = cv2.VideoCapture(0)
        self.web_cam_cap.set(3, self.frameWidth)
        self.web_cam_cap.set(4, self.frameHeight)
        print("Open Web Camera")

    def initialize_frame_cap(self):
        if self.use_robot:
            self.init_robot()
        else:
            self.init_web_cam()

    def get_frame(self):
        if self.use_robot:
            frame_reader = self.tello.get_frame_read()
            image = frame_reader.frame
        else:
            success, image = self.web_cam_cap.read()
        self.image_to_video = image
        return image

    def videoRecorder(self):
        # create a VideoWrite object, recoring to ./video.avi
        # 创建一个VideoWrite对象，存储画面至./video.avi
        height, width, _ = self.image_to_video.shape
        video = cv2.VideoWriter('../data/video/' + self.video_name, cv2.VideoWriter_fourcc(*'XVID'), 30,
                                (width, height))

        while self.keep_recording:
            video.write(self.image_to_video)
            time.sleep(1 / 30)

        video.release()