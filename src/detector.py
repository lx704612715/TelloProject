import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import itertools
import copy
import cvzone


class KeyPointClassifier(object):
    def __init__(
            self,
            model_path='models/gesture_recognizer.tflite',
            num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list


class Face_Detector:
    def __init__(self, Width, Height):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.frameWidth = Width
        self.frameHeight = Height

    def process_faces(self, frameRGB):
        results = self.face_detection.process(frameRGB)
        imageBGR_to_show = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2BGR)
        detected = False

        if results.detections:
            detected = True
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

            x = int(box_xmin * self.frameWidth)
            y = int(box_ymin * self.frameHeight)
            w = int(box_width * self.frameWidth)
            h = int(box_height * self.frameHeight)

            cv2.rectangle(imageBGR_to_show, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            cv2.circle(imageBGR_to_show, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            return detected, imageBGR_to_show, [[cx, cy], area]
        else:
            return detected, imageBGR_to_show, [[0, 0], 0]


class Gesture_Recognizer:
    def __init__(self, num_hands, mode_path, label_path, Width, Height):
        self.mpHands = mp.solutions.hands
        self.frameWidth = Width
        self.frameHeight = Height
        self.mp_hands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=num_hands, min_detection_confidence=0.7)
        self.results = None  # detected results from mediapipe
        self.detected_hands = []  # save detected hands involves box, landmarks
        self.mpDraw = mp.solutions.drawing_utils
        self.gesture_classifier = KeyPointClassifier(model_path=mode_path)
        # Load class names
        f = open(label_path, 'r')
        self.classNames = f.read().split('\n')
        f.close()
        x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
        y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        self.coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

    def process_hands(self, frameRGB, draw_landmarks=True, draw_box=True):
        self.detected_hands = []
        self.results = self.hands.process(frameRGB)
        # post process the result
        imageBGR_to_show = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2BGR)
        if self.results.multi_hand_landmarks:
            for handslms in self.results.multi_hand_landmarks:
                detected_hand = dict()
                hands_landmarks = []
                x_list = []
                y_list = []
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * self.frameWidth)
                    lmy = int(lm.y * self.frameHeight)
                    hands_landmarks.append([lmx, lmy])
                    x_list.append(lmx)
                    y_list.append(lmy)
                detected_hand["landmarks"] = hands_landmarks
                detected_hand["handslms"] = handslms
                # calculate bonding box
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                area = boxW * boxH
                detected_hand["bbox"] = bbox
                detected_hand["cen_bbox"] = [area, cx, cy]
                distCM = self.cal_hand_dist(image_BGR=imageBGR_to_show, hands_landmarks=hands_landmarks,
                                            xmin=xmin, ymin=ymin)
                detected_hand["distance"] = distCM
                self.detected_hands.append(detected_hand)
            # Drawing landmarks on frames
            for hand in self.detected_hands:
                print("length of detected hands", len(self.detected_hands))
                bbox = hand["bbox"]
                handslms = hand["handslms"]
                if draw_landmarks:
                    self.mpDraw.draw_landmarks(imageBGR_to_show, handslms, self.mpHands.HAND_CONNECTIONS)
                if draw_box:
                    cv2.rectangle(imageBGR_to_show, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
        return imageBGR_to_show, self.detected_hands

    def recognize_gesture(self, image_BGR):
        gesture_name = 'No Hand'
        for hand in self.detected_hands:
            hands_landmarks = hand["landmarks"]
            pre_landmark_list = self.gesture_classifier.pre_process_landmark(hands_landmarks)
            gesture_name = self.classNames[self.gesture_classifier.__call__(pre_landmark_list)]
            cv2.putText(image_BGR, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
        return image_BGR, gesture_name

    def cal_hand_dist(self, image_BGR, hands_landmarks, xmin, ymin):
        x1, y1 = hands_landmarks[5]
        x2, y2 = hands_landmarks[17]
        distance = int(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        distanceCM = self.coff[0] * distance ** 2 + self.coff[1] * distance + self.coff[2]
        cvzone.putTextRect(image_BGR, f'{int(distanceCM)} cm', (xmin + 5, ymin - 10), colorT=(255, 128, 0))
        return distanceCM
