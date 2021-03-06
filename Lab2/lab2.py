MAP_SIZE_PIXELS = 1000
MAP_SIZE_METERS = 40

import vrep
import sys
import time
import numpy as np
import math
from lib.map import Map
import matplotlib.pyplot as plt
import cv2


class Robot:
    client_id = -1
    e = 0
    prev_e = 0
    iSum = 0
    initial_speed = 0.6
    maintained_dist_right = 0.6
    prev_alpha = 0
    update_time = 0.01

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
        self.check_on_error(error_code, "Couldn't find left motor!", True)

        error_code, self.right_motor_handle = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_rightMotor',
                                                                       vrep.simx_opmode_oneshot_wait)
        self.check_on_error(error_code, "Couldn't find right motor!", True)

        error_code, self.laser_left = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_left',
                                                               vrep.simx_opmode_oneshot_wait)
        self.check_on_error(error_code, "Couldn't find left laser!", True)
        error_code, self.laser_right = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_right',
                                                                vrep.simx_opmode_oneshot_wait)
        self.check_on_error(error_code, "Couldn't find right laser!", True)
        error_code, self.laser_front = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_front',
                                                                vrep.simx_opmode_oneshot_wait)
        self.check_on_error(error_code, "Couldn't find front laser!", True)
        error_code, self.laser_middle = vrep.simxGetObjectHandle(self.client_id, 'Pr_sensor_middle',
                                                                 vrep.simx_opmode_oneshot_wait)
        self.check_on_error(error_code, "Couldn't find middle laser!", True)
        error_code, self.cuboid = vrep.simxGetObjectHandle(self.client_id, 'Cuboid',
                                                           vrep.simx_opmode_oneshot_wait)
        self.check_on_error(error_code, "Couldn't find cuboid!", True)

        self.relative_object = -1

        # looking for the robot
        error_code, self.robot_handle = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx',
                                                                 vrep.simx_opmode_oneshot_wait)
        self.check_on_error(error_code, "Couldn't find left robot!", True)

        self.prev_position = vrep.simxGetObjectPosition(self.client_id, self.robot_handle, self.cuboid,
                                                        vrep.simx_opmode_streaming)
        self.prev_orientation = vrep.simxGetObjectOrientation(self.client_id, self.robot_handle, self.relative_object,
                                                              vrep.simx_opmode_streaming)

        self.prev_position = \
        vrep.simxGetObjectPosition(self.client_id, self.robot_handle, self.cuboid, vrep.simx_opmode_buffer)[1]
        self.prev_orientation = \
        vrep.simxGetObjectOrientation(self.client_id, self.robot_handle, self.relative_object, vrep.simx_opmode_buffer)[
            1]

        e, self.lidar_handle1 = vrep.simxGetObjectHandle(self.client_id, 'fastHokuyo_sensor1',
                                                         vrep.simx_opmode_oneshot_wait)
        self.check_on_error(e, "Can not find lidar 1", True)
        e, self.lidar_handle2 = vrep.simxGetObjectHandle(self.client_id, 'fastHokuyo_sensor2',
                                                         vrep.simx_opmode_oneshot_wait)
        self.check_on_error(e, "Can not find lidar 2", True)
        self.prev_time = time.time()

        sec, msec = vrep.simxGetPingTime(self.client_id)
        print("Ping time: %f" % (sec + msec / 1000.0))
        print('Initialization is finished.')

    def check_on_error(self, e, msg, need_exit=False):
        if e != vrep.simx_return_ok:
            print('Error: {}'.format(msg))
            if need_exit:
                sys.exit()

    def set_motor_speed(self, left_speed, right_speed):
        e = vrep.simxSetJointTargetVelocity(self.client_id, self.left_motor_handle, left_speed,
                                            vrep.simx_opmode_oneshot_wait)
        self.check_on_error(e, "SetJointTargetVelocity for left motor got error code")
        e = vrep.simxSetJointTargetVelocity(self.client_id, self.right_motor_handle, right_speed,
                                            vrep.simx_opmode_oneshot_wait)
        self.check_on_error(e, "SetJointTargetVelocity for right motor got error code")
        # print("Motor's speed is set to {} {}".format(left_speed, right_speed))

    def get_lidar_data(self):
        point_data, dist_data = [], []

        e, buf_dist = vrep.simxGetStringSignal(self.client_id, "distances", vrep.simx_opmode_buffer)
        e, buf_measured = vrep.simxGetStringSignal(self.client_id, "lidarMeasuredData", vrep.simx_opmode_buffer)
        self.check_on_error(e, "simxGetStringSignal lidar distances error")
        if len(buf_measured) > 2 and len(buf_dist) > 2:
            dist_data = vrep.simxUnpackFloats(buf_dist)
            measuredData = vrep.simxUnpackFloats(buf_measured)
            point_data = np.reshape(measuredData, (int(len(measuredData) / 3), 3))
        return e, point_data, dist_data

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

    def calc_odometry(self):
        curr_position = \
        vrep.simxGetObjectPosition(self.client_id, self.robot_handle, self.cuboid, vrep.simx_opmode_buffer)[1]
        curr_orientation = \
        vrep.simxGetObjectOrientation(self.client_id, self.robot_handle, self.relative_object, vrep.simx_opmode_buffer)[
            1]
        # print(curr_position)
        delta_x = self.prev_position[0] - curr_position[0]
        delta_y = self.prev_position[1] - curr_position[1]
        if delta_y > 0:
            alpha = math.degrees(math.atan(delta_x / delta_y))
        elif delta_y < 0:
            alpha = 180 - math.degrees(math.atan(delta_x / -delta_y))
        else:
            alpha = self.prev_alpha
        print("alpha: {}, z-orien: {}".format(alpha, math.degrees(curr_orientation[2]) + 90))
        alpha = math.degrees(curr_orientation[2]) + 90
        self.prev_position = curr_position
        self.prev_orientation = curr_orientation
        self.prev_alpha = alpha
        curr_position[0] *= 1
        curr_position[1] *= -1
        return curr_position, alpha

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

        e, data1 = vrep.simxGetStringSignal(self.client_id, "lidarMeasuredData", vrep.simx_opmode_streaming)
        e, data2 = vrep.simxGetStringSignal(self.client_id, "distances", vrep.simx_opmode_streaming)
        self.check_on_error(e, "simxGetStringSignal lidar error")

        map = Map(MAP_SIZE_PIXELS, MAP_SIZE_PIXELS)

        prev_dist = 0
        err_arr_tmp = np.zeros(100)
        left_arr_tmp, right_arr_tmp = np.zeros(100), np.zeros(100)
        err_arr = []
        left_arr, right_arr = [], []
        counter = 0

        self.prev_position = \
        vrep.simxGetObjectPosition(self.client_id, self.robot_handle, self.cuboid, vrep.simx_opmode_buffer)[1]
        self.prev_orientation = \
        vrep.simxGetObjectOrientation(self.client_id, self.robot_handle, self.relative_object, vrep.simx_opmode_buffer)[
            1]
        skip = 0
        while vrep.simxGetConnectionId(self.client_id) != -1:
            front_detection, left_detection, middle_detection, right_detection, distance_front, distance_left, distance_middle, distance_right = self.get_proximity_data()
            if left_detection and right_detection:
                dist = min(self.calc_dist(distance_left, distance_right), distance_middle)
            elif not left_detection and not right_detection and middle_detection:
                dist = distance_middle
            elif not left_detection:
                # print("Turning")
                dist = 0.9
                if middle_detection:
                    dist = min(dist, distance_middle, distance_right)
            elif left_detection and middle_detection:
                dist = min(distance_left, distance_middle)

            if front_detection:
                dist = min(distance_front, dist)
            # print("Distance: {}".format(dist))

            err_arr_tmp[counter] = self.maintained_dist_right - dist
            self.fix_distance(dist, prev_dist)

            e, lidar_data, dist_data = self.get_lidar_data()
            point_data_len = len(lidar_data)
            dist_data_len = len(dist_data)
            dist_data = dist_data[::-1]
            slam_dist_data = [i * 1 for i in dist_data[0:-2]]
            print(len(slam_dist_data))

            # getting robot position data and updating the map
            pos, angle = self.calc_odometry()
            map.update(pos, angle, slam_dist_data)

            time.sleep(self.update_time)

            prev_dist = dist

        vrep.simxFinish(self.client_id)
        return err_arr, left_arr, right_arr

    def get_proximity_data(self):
        (e, left_detection, left, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_left,
                                                                     vrep.simx_opmode_buffer)
        self.check_on_error(e, "Left sensor data reading error")

        (e, right_detection, right, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_right,
                                                                     vrep.simx_opmode_buffer)
        self.check_on_error(e, "Right sensor data reading error")

        (e, front_detection, front, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_front,
                                                                     vrep.simx_opmode_buffer)
        self.check_on_error(e, "Front sensor data reading error")

        (e, middle_detection, middle, detectedObjectHandle,
         detectedSurfaceNormalVector) = vrep.simxReadProximitySensor(self.client_id, self.laser_middle,
                                                                     vrep.simx_opmode_buffer)
        self.check_on_error(e, "Middle sensor data reading error")

        return front_detection, left_detection, middle_detection, right_detection, np.linalg.norm(
            front), np.linalg.norm(left), np.linalg.norm(middle), np.linalg.norm(right)


if __name__ == '__main__':
    robot = Robot()
    err_arr, left, right = robot.start_simulation()
    # x = [i + 1 for i in range(len(err_arr))]
    #
    #
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(x, err_arr)
    # plt.title('Error change')
    # plt.ylabel('Error')
    # plt.xlabel('Time, sec')
    # plt.grid(True)
    #
    # plt.subplot(212)
    # plt.plot(x, left, x, right)
    # plt.title('Velocity change (blue - left, orange - right)')
    # plt.ylabel('Velocity')
    # plt.xlabel('Time, sec')
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()
    print('Done')
