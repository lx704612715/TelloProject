import cv2
import numpy as np
import cvzone
from frame_receiver import Frame_Receiver
from controller import Tello_Controller


class InteractiveTracker(object):
    def __init__(self, frameWidth, frameHeight, detection_threshold=0.2):
        self.test = None
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.image = np.zeros((frameHeight, frameWidth, 3))

        # cropping parameters
        self.crop_image_pos = []  # [x_start, y_start, x_end, y_end]
        self.cropping = None

        # detection parameters
        self.detection_threshold = detection_threshold

        # dnn detection parameter
        # self.weightsPath = '../models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        # self.configPath = "../models/frozen_inference_graph.pb"
        # self.net = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        # self.net.setInputSize((320, 320))
        # self.net.setInputScale(1.0 / 127.5)
        # self.net.setInputMean((127.5, 127.5, 127.5))
        # self.net.setInputSwapRB(True)
        # self.thres = 0.55
        # self.nmsThres = 0.2
        # self.classNames = []
        # classFile = '../models/coco.names'
        # with open(classFile, 'rt') as f:
        #     self.classNames = f.read().split('\n')

        # template matching detector parameter
        self.template_image = None

        # object tracker parameters
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking_mode = False
        self.id = 0

    def set_mouse_callback(self, window_name="image"):
        cv2.setMouseCallback(window_name, self.mouse_crop)
        # cv2.setMouseCallback(window_name, self.select_ROI)

    def select_ROI(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox = cv2.selectROI("image", self.image)
            self.template_image = self.image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            self.crop_image_pos = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            self.tracker.init(self.image, bbox)
            self.tracking_mode = True
            cv2.imshow("Cropped", self.template_image)

    def mouse_crop(self, event, x, y, flags, param):
        # if the left mouse button was DOWN, start RECORDING
        # (x, y) coordinates and indicate that cropping is being
        if event == cv2.EVENT_LBUTTONDOWN:
            self.template_image = None
            self.crop_image_pos = [x, y, x, y]
            self.cropping = True

        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                self.crop_image_pos[2], self.crop_image_pos[3] = x, y

        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates
            self.crop_image_pos[2], self.crop_image_pos[3] = x, y
            self.cropping = False  # cropping is finished

            refPoint = [(self.crop_image_pos[0], self.crop_image_pos[1]),
                        (self.crop_image_pos[2], self.crop_image_pos[3])]
            if refPoint[0][0] != refPoint[1][0] and refPoint[0][1] != refPoint[1][
                1]:  # when two different points were found
                self.template_image = self.image[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                w = self.crop_image_pos[2] - self.crop_image_pos[0]
                h = self.crop_image_pos[3] - self.crop_image_pos[1]
                bbox = [self.crop_image_pos[0], self.crop_image_pos[1], w, h]
                self.tracker.init(self.image, bbox)
                self.tracking_mode = True
                self.id += 1
                cv2.imshow("Cropped", self.template_image)

    def template_matching_detection(self, image_bgr):
        if self.template_image is not None:
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            template_image_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(image_gray, template_image_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= self.detection_threshold:
                h, w = self.template_image.shape[:2]
                cv2.rectangle(image_bgr, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 255, 255), 2)
                bbox = [max_loc[0], max_loc[1], w, h]
                self.tracker.init(self.image, bbox)
            # loc = np.where(res >= self.detection_threshold)
            # for pt in zip(*loc[::-1]):
            #     h, w = self.template_image.shape[:2]
            #     cv2.rectangle(image_bgr, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        return image_bgr

    def dnn_detection(self, image_bgr):
        classIds, confs, bbox = self.net.detect(image_bgr, confThreshold=self.thres, nmsThreshold=self.nmsThres)
        try:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cvzone.cornerRect(image_bgr, box)
                cv2.putText(image_bgr, f'{self.classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                            (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 255, 0), 2)
        except Exception as e:
            print(e)
            pass
        return image_bgr

    def tracking(self, image_bgr):
        success = False
        curt_x = 0
        curt_y = 0
        if self.template_image is not None and self.tracking_mode:
            success, bbox = self.tracker.update(image_bgr)
            # Draw bounding box
            if success:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 2)
                cvzone.putTextRect(image, "ID"+str(self.id), (bbox[0] + 5, bbox[1] - 10), colorT=(255, 128, 0), scale=2, thickness=3)
                cv2.putText(image, " Tracking Success", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 170, 50), 2, cv2.LINE_AA)
                curt_x = bbox[0] + int(bbox[2]//2)
                curt_y = bbox[1] + int(bbox[3]//2)
                cv2.circle(image, (curt_x, curt_y), 5, (0, 0, 255), 2)
            else:
                # Tracking failure
                cv2.putText(image, "Tracking failure detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                self.re_detection(image)
        return image, success, curt_x, curt_y

    def re_detection(self, image):
        """ if tracking fails, re detection will be executed once to re-locate object using template matching method
        """
        print("running re detection")
        image = self.template_matching_detection(image_bgr=image)


if __name__ == "__main__":
    Use_Drone = True
    Control_Drone = True
    Record_Video = False

    frameWidth = 640
    frameHeight = 480

    frame_receiver = Frame_Receiver(use_robot=Use_Drone, Width=frameWidth, Height=frameHeight,
                                    record_video=Record_Video, video_name="tracking2.avi")

    inter_detector = InteractiveTracker(frameWidth=frameWidth, frameHeight=frameHeight, detection_threshold=0.8)
    cv2.namedWindow('image')
    inter_detector.set_mouse_callback(window_name="image")

    kp = [0.15, 0.15, 0.8]
    kv = [-0.05, -0.05, -0.8]
    controller = Tello_Controller(kp, kv, frame_receiver.tello, send_command=Control_Drone, online_plot=False)
    controller.set_d_Visual_Servo_Goal(d_x=frameWidth // 2, d_y=frameHeight // 2, d_area=[25000, 40000], d_dist=0)

    video_image_list = []
    if Control_Drone:
        controller.robot.takeoff()
        controller.robot.send_rc_control(0, 0, 0, 0)

    try:
        while True:
            image = frame_receiver.get_frame()
            inter_detector.image = image.copy()
            if inter_detector.cropping:
                x_start, y_start, x_end, y_end = inter_detector.crop_image_pos
                cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

            if inter_detector.template_image is not None:
                timer = cv2.getTickCount()
                image, success, curt_x, curt_y = inter_detector.tracking(image)
                if success:
                    controller.track_box(curt_x=curt_x, curt_y=curt_y, curt_dist=controller.d_dist)
                else:
                    controller.static_mode()
                    controller.control_mode = "static"
                controlMode = "Control Mode: " + str(controller.control_mode)
                cv2.putText(image, controlMode, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                # Display FPS on frame
                cv2.putText(image, "FPS : " + str(int(fps)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 170, 50), 2, cv2.LINE_AA)
                frame_receiver.image_to_video = np.copy(image)
                # video_image_list.append(image)
            else:
                controller.static_mode()
            video_image_list.append(image)
            cv2.imshow("image", image)
            if cv2.waitKey(1) == ord('q'):

                height, width, _ = image.shape
                video = cv2.VideoWriter('../data/video/' + "test.avi", cv2.VideoWriter_fourcc(*'XVID'), 15,
                                        (width, height))
                for image in video_image_list:
                    video.write(image)
                video.release()

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
