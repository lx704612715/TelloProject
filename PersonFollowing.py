import cv2
import numpy as np
from djitellopy import tello
import time
import mediapipe as mp
from threading import Thread
from gtts import gTTS
import speech_recognition as sr
import time
from pygame import mixer  # Load the popular external library
mixer.init()

def findFace(img):

    img.flags.writeable = False
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    if results.detections:
        best_score = 0
        best_detection = 0
        for i, detection in enumerate(results.detections):
            if detection.score[0] >= best_score:
                best_score = detection.score[0]
                best_detection = i
        best_location = results.detections[best_detection].location_data.relative_bounding_box

        box_xmin = best_location.xmin
        box_ymin = best_location.ymin
        box_width = best_location.width
        box_height = best_location.height

        x = int(box_xmin * frameWidth)
        y = int(box_ymin * frameHeight)
        w = int(box_width * frameWidth)
        h = int(box_height * frameHeight)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        return img, [[cx, cy], area]

    else:
        return img, [[0, 0], 0]

def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0
    error = x - w//2
    speed = pid[0] * error + pid[1] * (error-pError)
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    fb, speed = limited_vel(fb, speed, max_ud=20, max_yaw=50)
    print("Yaw speed", speed)
    print("forward speed", fb)
    if flying:
        me.send_rc_control(0, fb, 0, speed)
    return error

def limited_vel(ud_vel, yaw_vel, max_ud=30, max_yaw=80):

    if ud_vel > max_ud:
        ud_vel = max_ud
    elif ud_vel < -max_ud:
        ud_vel = -max_ud
    elif ud_vel < 10 and ud_vel > -10:
        ud_vel = 0

    if yaw_vel > max_yaw:
        yaw_vel = max_yaw
    elif yaw_vel < -max_yaw:
        yaw_vel = -max_yaw
    elif yaw_vel < 2 and yaw_vel > -2:
        yaw_vel = 0

    return ud_vel, yaw_vel

def videoRecorder():
    # create a VideoWrite object, recoring to ./video.avi
    # 创建一个VideoWrite对象，存储画面至./video.avi
    height, width, _ = img.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(img)
        time.sleep(1 / 30)

    video.release()

def speek(text, lang, filename="voice1.mp3"):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    mixer.music.load(filename)
    mixer.music.play()
    while mixer.music.get_busy():  # wait for music to finish playing
        time.sleep(1)

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak")
        audio = r.listen(source, phrase_time_limit=4)
        said = ""
        try:
            said = r.recognize_google(audio)
        except Exception as e:
            print("Error: ", e)
    return said

me = tello.Tello()
me.connect()
print("Battery level is: ", me.get_battery())
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
flying = True
startCount = 0
frameWidth = 640
frameHeight = 480
me.streamon()
global img
img = me.get_frame_read().frame
img = cv2.resize(img, (frameWidth, frameHeight))
IfVideoRecording = True
keepRecording = True

if flying:
    input("Start flying!!")
    me.takeoff()
    # me.send_rc_control(0, 0, 60, 0)
    me.move_up(100)
    time.sleep(1)
    me.send_rc_control(0, 0, 0, 0)
    startCount = 1

fbRange = [8000, 30000]
pid = [0.3, 0.4, 0]
pError = 0

if IfVideoRecording:
    recorder = Thread(target=videoRecorder)
    recorder.start()
while True:
    try:
        img = me.get_frame_read().frame
        img = cv2.resize(img, (frameWidth, frameHeight))
        img, info = findFace(img)
        pError = trackFace(info, frameWidth, pid, pError)
        print("Center", info[0], "Area", info[1])
        cv2.imshow("Output", img)
        print("sending command")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            me.land()
            print("finishing")

            if IfVideoRecording:
                keepRecording = False
                recorder.join()
                cv2.destroyAllWindows()
            break

    except Exception as e:
        pass
        # me.land()

