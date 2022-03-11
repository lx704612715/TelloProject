import numpy as np
import matplotlib.pyplot as plt
# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    label = ["fb", "ud", "yaw"]
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        ax.set_ylim(-50, 50)
        # create a variable for the line so we can later update it
        line1 = ax.plot(x_vec, y1_data, '-o', alpha=0.8, label=label)
        # update plot label/title
        plt.legend()
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1[0].set_ydata(y1_data[:, 0])
    line1[1].set_ydata(y1_data[:, 1])
    line1[2].set_ydata(y1_data[:, 2])
    # adjust limits if new data goes beyond bounds
    # if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
    #     plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1


class Tello_Controller(object):
    def __init__(self, kp, kv, robot, send_command, online_plot=False):
        self.kp = kp  # kp in x, y, area
        self.kv = kv  # kv in x, y, area
        self.robot = robot  # tello robot class
        self.send_command = send_command  # if sends command to robot
        self.last_error = [0, 0, 0]  # last error in x, y, area
        self.d_x = None  # desired image x pixel value
        self.d_y = None  # desired image y pixel value
        self.d_area = None  # [min_area, max_area]
        self.d_dist = None  # desired distance
        self.history_size = 10
        self.history_command = np.zeros([self.history_size, 3])
        self.control_mode = "tracking"  # control mode ["tracking", "flip", "pointing", "stationary"]

        # for online plot
        self.online_plot = online_plot
        self.line1 = []
        # if flip counter larger than 10, execute a flip maneuver
        self.flip_counter = 0
        # if point counter larger than 10, moving to the point direction
        self.point_counter = 0
        # if point counter larger than 100, moving to the point direction
        self.land_counter = 0

    def set_d_Visual_Servo_Goal(self, d_x, d_y, d_area, d_dist):
        self.d_x = d_x
        self.d_y = d_y
        self.d_area = d_area
        self.d_dist = d_dist

    def static_mode(self):
        # print("Sending Command: fb: {}, up: {}, yaw: {}".format(0, 0, 0))

        self.control_mode = "stationary"
        if self.send_command:
            self.robot.send_rc_control(0, 0, 0, 0)
            self.last_error = [0, 0, 0]

    def track_box(self, curt_x, curt_y, curt_dist):
        assert self.d_y is not None, "You have to set desired image target"

        self.control_mode = "tracking"
        error_x = curt_x - self.d_x
        error_y = -1 * (curt_y - self.d_y)
        error_dist = curt_dist - self.d_dist
        speed_yaw = self.kp[0] * error_x + self.kv[0] * (error_x - self.last_error[0])
        speed_up_down = self.kp[1] * error_y + self.kv[1] * (error_y - self.last_error[1])
        speed_fb = self.kp[2] * error_dist + self.kv[2] * (error_dist - self.last_error[2])
        self.last_error = [error_x, error_y, error_dist]

        speed_up_down = int(np.clip(speed_up_down, -50, 50))
        speed_yaw = int(np.clip(speed_yaw, -50, 50))
        speed_fb = int(np.clip(speed_fb, -30, 30))

        command = np.array([speed_fb, speed_up_down, speed_yaw]).reshape(1, 3)
        self.history_command = np.vstack([self.history_command[1:, :], command])
        x_vec = np.linspace(0, 1, self.history_size + 1)[0:-1]

        if self.online_plot:
            self.line1 = live_plotter(x_vec, self.history_command, self.line1)

        if self.send_command:
            self.robot.send_rc_control(0, speed_fb, speed_up_down, speed_yaw)
            print("Sending Command: fb: {}, up: {}, yaw: {}".format(speed_fb, speed_up_down, speed_yaw))

    def gesture_control(self, gesture_name, hands):

        if len(hands) == 0:
            print("No hands")
            self.static_mode()
            self.rest_counter()
            return 0

        curt_x = hands[0]["cen_bbox"][1]
        curt_y = hands[0]["cen_bbox"][2]
        curt_dist = hands[0]["distance"]
        hands_landmarks = hands[0]["landmarks"]
        # index_finger_tip = hands_landmarks[8]
        # wrist = hands_landmarks[0]

        if gesture_name == "Open":
            print("Tracking mode")
            self.track_box(curt_x=curt_x, curt_y=curt_y, curt_dist=curt_dist)
            self.rest_counter()
        elif gesture_name == "Close":
            print("Flip mode")
            self.flip_counter += 1
            if self.flip_counter >= 10:
                self.control_mode = "flip"
                if self.send_command:
                    self.robot.flip_back()
                    self.robot.send_rc_control(0, 0, 0, 0)
                    self.rest_counter()
        elif gesture_name == "Pointer":
            print("Moving up mode")
            self.control_mode = "pointer"
            self.point_counter += 1
            if self.point_counter >= 10:
                if self.send_command:
                    self.robot.move_up(30)
                    self.rest_counter()
        elif gesture_name == "OK":
            print("Landing")
            self.control_mode = "land"
            self.land_counter += 1
            if self.land_counter >= 70:
                if self.send_command:
                    self.robot.land()
        else:
            print("Unknown Gesture")
            self.static_mode()
            self.rest_counter()

    def rest_counter(self):
        self.flip_counter = 0
        self.point_counter = 0
        self.land_counter = 0
