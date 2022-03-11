import cv2
from controller import Tello_Controller
from detector import Gesture_Recognizer, Face_Detector
from frame_receiver import Frame_Receiver


if __name__ == "__main__":

    Use_Drone = True
    Control_Drone = True
    Record_Video = True

    frameWidth = 640
    frameHeight = 480

    frame_receiver = Frame_Receiver(use_robot=Use_Drone, Width=frameWidth, Height=frameHeight,
                                    record_video=Record_Video, video_name="test.avi")

    kp = [0.3, 0.3, 0.8]
    kv = [-0.05, -0.05, -0.8]
    controller = Tello_Controller(kp, kv, frame_receiver.tello, send_command=Control_Drone, online_plot=False)
    controller.set_d_Visual_Servo_Goal(d_x=frameWidth // 2, d_y=frameHeight // 2, d_area=[25000, 40000], d_dist=80)

    if Control_Drone:
        controller.robot.takeoff()
        controller.robot.send_rc_control(0, 0, 0, 0)

    gesture_recognizer = Gesture_Recognizer(num_hands=1, mode_path="../models/gesture_recognizer.tflite",
                                            label_path="../models/gesture.names", Width=frameWidth,
                                            Height=frameHeight)

    # face_detector = Face_Detector(Width=frameWidth, Height=frameHeight)

    try:
        while True:
            image = frame_receiver.get_frame()
            image = cv2.resize(image, (frameWidth, frameHeight))
            # Flip the frame vertically
            frameRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_BGR, detected_hands = gesture_recognizer.process_hands(frameRGB)
            # detected, imageBGR_to_show, predicted_result = face_detector.process_faces(frameRGB)
            # we only track the first detected hand
            image_BGR, gesture_name = gesture_recognizer.recognize_gesture(image_BGR)
            cv2.putText(image_BGR, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            controller.gesture_control(gesture_name=gesture_name, hands=gesture_recognizer.detected_hands)
            controlMode = "Control Mode: " + str(controller.control_mode)
            cv2.putText(image_BGR, controlMode, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

            frame_receiver.image_to_video = image_BGR
            cv2.imshow("Output", image_BGR)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                if Use_Drone:
                    controller.robot.land()
                if Record_Video:
                    frame_receiver.keep_recording = False
                    frame_receiver.recorder.join()
                break
    except Exception as e:
        print(e)
    finally:
        if Use_Drone:
            controller.robot.land()
