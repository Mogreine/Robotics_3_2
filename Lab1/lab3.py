from matplotlib.ticker import NullFormatter

import vrep
import sys
import time
import numpy as np
import math
import cv2
import imutils
import matplotlib.pyplot as plt


class Robot:
    client_id = -1
    e = 0
    prev_e = 0
    iSum = 0
    initial_speed = 0.6
    maintained_dist_right = 0.4
    cylinders_pos = list()

    def __init__(self):
        vrep.simxFinish(-1)
        self.client_id = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        if self.client_id != -1:
            print("Connected to remote server")
        else:
            print('Connection not successful')
            sys.exit('Could not connect')

        self.left_speed, self.right_speed = self.initial_speed, self.initial_speed

        error_code, self.left_motor_handle = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_leftMotor',
                                                                      vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find left motor!", True)

        error_code, self.right_motor_handle = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_rightMotor',
                                                                       vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find right motor!", True)

        error_code, self.laser_left = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_left',
                                                               vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find left laser!", True)
        error_code, self.laser_right = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_right',
                                                                vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find right laser!", True)
        error_code, self.laser_front = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_front',
                                                                vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find front laser!", True)
        error_code, self.laser_middle = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_middle',
                                                                 vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find middle laser!", True)
        error_code, self.cuboid = vrep.simxGetObjectHandle(self.client_id, 'Cuboid',
                                                                 vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find cuboid!", True)
        error_code, self.robot_handle = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx',
                                                                 vrep.simx_opmode_oneshot_wait)
        self.check_error(error_code, "Couldn't find left robot!", True)

        self.prev_position = vrep.simxGetObjectPosition(self.client_id, self.robot_handle, self.cuboid,
                                                        vrep.simx_opmode_streaming)
        self.prev_orientation = vrep.simxGetObjectOrientation(self.client_id, self.robot_handle, -1,
                                                              vrep.simx_opmode_streaming)

        self.image = cv2.imread('map.png')

        sec, msec = vrep.simxGetPingTime(self.client_id)
        print("Ping time: %f" % (sec + msec / 1000.0))
        print('Initialization is finished.')

    def check_error(self, e, msg, need_exit=False):
        if e != vrep.simx_return_ok:
            print(f'Error: {msg}')
            if need_exit:
                sys.exit()

    def set_motor_speed(self, left_speed, right_speed):
        e = vrep.simxSetJointTargetVelocity(self.client_id, self.left_motor_handle, left_speed,
                                            vrep.simx_opmode_oneshot_wait)
        self.check_error(e, "SetJointTargetVelocity for left motor got error code")
        e = vrep.simxSetJointTargetVelocity(self.client_id, self.right_motor_handle, right_speed,
                                            vrep.simx_opmode_oneshot_wait)
        self.check_error(e, "SetJointTargetVelocity for right motor got error code")
        # print(f"Motor's speed is set to {left_speed} {right_speed}")

    def fix_distance(self, dist, old_dist):
        # constants
        kp = 2
        kd = 1
        ki = 0.1
        iMin, iMax = -0.2, 0.2

        # error calculation
        self.e = self.maintained_dist_right - dist

        # Prop
        up = kp * self.e

        # Diff
        ud = kd * (dist - old_dist)

        # Integral
        self.iSum += self.e
        self.iSum = max(iMin, self.iSum)
        self.iSum = min(iMax, self.iSum)
        ui = ki * self.iSum

        res = up + ud + ui
        self.set_motor_speed(self.left_speed - res, self.right_speed + res)
        # self.robot.change_velocity([self.left_speed - res, self.right_speed + res])
        self.prev_e = self.e
        return self.left_speed - res, self.right_speed + res

    def calc_dist(self, left_dist, right_dist):
        hyp = math.sqrt(left_dist ** 2 + right_dist ** 2)
        dist = left_dist * right_dist / hyp
        return dist

    def detect_cylinder(self, contour):
        per = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * per, True)

        # looking at vertices number, if 4 - it's either a rectangle or a cylinder
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ratio = w / h

            if 0.90 <= ratio <= 1.1:
                return -1
            else:
                return x + h // 2
        else:
            return -1

    def is_cylinder_unique(self, pos):
        for c_pos in self.cylinders_pos:
            if (pos[0] - c_pos[0]) ** 2 + (pos[1] - c_pos[1]) ** 2 <= c_pos[2] ** 2:
                return False

        return True

    def process_image(self, img):
        img_resized = imutils.resize(img, width=300)
        ratio = img.shape[0] / img_resized.shape[0]

        # gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # thresh = cv2.threshold(blurred, 150, 200, cv2.THRESH_BINARY)[1]
        # cv2.imshow('thresh', thresh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        hsv_im = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        lower_green = np.array([10, 100, 100])
        upper_green = np.array([60, 255, 255])

        mask = cv2.inRange(hsv_im, lower_green, upper_green)
        # masked_im = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        # cv2.imshow('mask', mask)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            cylinder_coords = self.detect_cylinder(contour)

            if cylinder_coords != -1:
                contour = contour.astype('float') * ratio
                contour = contour.astype('int')
                cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)

                # calculating an approximate position
                # angle = math.radians((256 - cylinder_coords) // (512 / 60))
                # r = 5
                # y0 = mine.y + math.cos(angle) * r
                # x0 = mine.x + math.sin(angle) * r
                scaling = 20
                dev_p = 256 - cylinder_coords
                screen_width = 2 * 5 * math.tan(math.radians(30))
                coef = screen_width / 512
                dev = coef * dev_p * scaling
                robot_pos = vrep.simxGetObjectPosition(self.client_id, self.robot_handle, self.cuboid,
                                                       vrep.simx_opmode_buffer)[1]
                robot_pos = [p * scaling for p in robot_pos]
                robot_pos[1] *= -1
                # robot_pos[0], robot_pos[1] = robot_pos[1], robot_pos[0]
                robot_pos[0] += 500
                robot_pos[1] += 200
                robot_orientation = vrep.simxGetObjectOrientation(self.client_id, self.robot_handle, -1,
                                                       vrep.simx_opmode_buffer)[1][2] + math.radians(90)
                angle = (robot_orientation + math.radians(dev_p / 8.53))
                dist = 5 * scaling
                pos = (robot_pos[0] + dist * math.sin(angle), robot_pos[1] + dist * math.cos(angle))
                print(pos)
                print(math.degrees(angle))
                pos = (int(pos[0]), int(pos[1]))
                if self.is_cylinder_unique(pos):
                    self.cylinders_pos.append((pos[0], pos[1], 3 * scaling))
                    cv2.circle(self.image, pos, 5, (0, 0, 255), cv2.FILLED)

                cv2.imshow('SLAM', self.image)
                k = cv2.waitKey(1) & 0xFF
                # saving the map if 's' is pressed
                if k == ord('s'):
                    cv2.imwrite('map2.png', self.image)


        return img

    def start_simulation(self):
        self.set_motor_speed(self.left_speed, self.right_speed)

        (errorCode, detectionState, detectedPoint, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_left,
                                                                     vrep.simx_opmode_streaming)
        (errorCode, detectionState, detectedPoint, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_right,
                                                                     vrep.simx_opmode_streaming)
        (errorCode, detectionState, detectedPoint, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_front,
                                                                     vrep.simx_opmode_streaming)
        (errorCode, detectionState, detectedPoint, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_middle,
                                                                     vrep.simx_opmode_streaming)

        prev_dist = 0

        res, v0 = vrep.simxGetObjectHandle(self.client_id, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)
        res, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, v0, 0, vrep.simx_opmode_streaming)

        while vrep.simxGetConnectionId(self.client_id) != -1:
            front_detection, left_detection, middle_detection, right_detection, distance_front, distance_left, distance_middle, distance_right = self.get_proximity_data()
            if left_detection and right_detection:
                dist = min(self.calc_dist(distance_left, distance_right), distance_middle)
            elif not left_detection and not right_detection and middle_detection:
                dist = distance_middle
            elif not left_detection:
                dist = 0.7
                if middle_detection:
                    dist = min(dist, distance_middle, distance_right)
            elif left_detection and middle_detection:
                dist = min(distance_left, distance_middle)

            if front_detection:
                dist = min(distance_front / 2, dist)
            # print("Distance: {}".format(dist))

            self.fix_distance(dist, prev_dist)

            # if counter == 100:
            #     counter = 0
            #     err_arr.append(err_arr_tmp.mean())
            #     left_arr.append(left_arr_tmp.mean())
            #     right_arr.append(right_arr_tmp.mean())
            #     left_arr_tmp, right_arr_tmp, err_arr_tmp = np.zeros(100), np.zeros(100), np.zeros(100)

            # camera picture drawing
            err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, v0, 0, vrep.simx_opmode_buffer)
            if err == vrep.simx_return_ok:
                # processing the image
                img = np.array(image, dtype=np.uint8)
                img.resize([resolution[1], resolution[0], 3])
                img = cv2.flip(img, 0)

                img = self.process_image(img)
                print(f"Опознано цилиндров: {len(self.cylinders_pos)}")

                cv2.imshow('vision sensor', img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    cv2.imwrite('camera.png', img)

            elif err == vrep.simx_return_novalue_flag:
                print("no image yet")
                pass
            else:
                print(err)

            prev_dist = dist

            time.sleep(0.01)

        vrep.simxFinish(self.client_id)

    def get_proximity_data(self):
        (e, left_detection, left, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_left,
                                                                     vrep.simx_opmode_buffer)
        self.check_error(e, "Left sensor data reading error")

        (e, right_detection, right, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_right,
                                                                     vrep.simx_opmode_buffer)
        self.check_error(e, "Right sensor data reading error")

        (e, front_detection, front, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_front,
                                                                     vrep.simx_opmode_buffer)
        self.check_error(e, "Front sensor data reading error")

        (e, middle_detection, middle, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_middle,
                                                                     vrep.simx_opmode_buffer)
        self.check_error(e, "Middle sensor data reading error")

        return front_detection, left_detection, middle_detection, right_detection, np.linalg.norm(
            front), np.linalg.norm(left), np.linalg.norm(middle), np.linalg.norm(right)


if __name__ == '__main__':
    robot = Robot()
    robot.start_simulation()

    print('Done')
