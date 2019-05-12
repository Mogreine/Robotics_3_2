from matplotlib.ticker import NullFormatter

import vrep
import sys
import time
import numpy as np
import math
import matplotlib.pyplot as plt


class Robot:
    client_id = -1
    e = 0
    prev_e = 0
    iSum = 0
    initial_speed = 0.5
    maintained_dist_right = 0.6

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

        sec, msec = vrep.simxGetPingTime(self.client_id)
        print("Ping time: %f" % (sec + msec / 1000.0))
        print('Initialization is finished.')

    def check_on_error(self, e, msg, need_exit=False):
        if e != vrep.simx_return_ok:
            print(f'Error: {msg}')
            if need_exit:
                sys.exit()

    def set_motor_speed(self, left_speed, right_speed):
        e = vrep.simxSetJointTargetVelocity(self.client_id, self.left_motor_handle, left_speed,
                                            vrep.simx_opmode_oneshot_wait)
        self.check_on_error(e, "SetJointTargetVelocity for left motor got error code")
        e = vrep.simxSetJointTargetVelocity(self.client_id, self.right_motor_handle, right_speed,
                                            vrep.simx_opmode_oneshot_wait)
        self.check_on_error(e, "SetJointTargetVelocity for right motor got error code")
        print(f"Motor's speed is set to {left_speed} {right_speed}")

    def fix_distance(self, dist, old_dist):
        # constants
        kp = 0.4
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
        self.prev_e = self.e
        return self.left_speed - res, self.right_speed + res

    def calc_dist(self, left_dist, right_dist):
        hyp = math.sqrt(left_dist ** 2 + right_dist ** 2)
        dist = left_dist * right_dist / hyp
        return dist

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
        prev_left_detection = True
        err_arr_tmp = np.zeros(100)
        left_arr_tmp, right_arr_tmp = np.zeros(100), np.zeros(100)
        err_arr = []
        left_arr, right_arr = [], []
        counter = 0
        while vrep.simxGetConnectionId(self.client_id) != -1:
            front_detection, left_detection, middle_detection, right_detection, distance_front, distance_left, distance_middle, distance_right = self.get_proximity_data()
            if left_detection and right_detection:
                dist = min(self.calc_dist(distance_left, distance_right), distance_middle)
            elif not left_detection and not right_detection and middle_detection:
                dist = distance_middle
            elif not left_detection:
                print("Turning")
                dist = 0.7
                if middle_detection:
                    dist = min(dist, distance_middle, distance_right)
            elif left_detection and middle_detection:
                    dist = min(distance_left, distance_middle)

            if front_detection:
                dist = min(distance_front, dist)
            print("Distance: {}".format(dist))

            err_arr_tmp[counter] = self.maintained_dist_right - dist
            left_arr_tmp[counter], right_arr_tmp[counter] = self.fix_distance(dist, prev_dist)

            counter += 1
            if counter == 100:
                counter = 0
                err_arr.append(err_arr_tmp.mean())
                left_arr.append(left_arr_tmp.mean())
                right_arr.append(right_arr_tmp.mean())
                left_arr_tmp, right_arr_tmp, err_arr_tmp = np.zeros(100), np.zeros(100), np.zeros(100)

            prev_dist = dist

            time.sleep(0.01)

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

        return front_detection, left_detection, middle_detection, right_detection, np.linalg.norm(front), np.linalg.norm(left), np.linalg.norm(middle), np.linalg.norm(right)


if __name__ == '__main__':
    robot = Robot()
    err_arr, left, right = robot.start_simulation()
    x = [i + 1 for i in range(len(err_arr))]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, err_arr)
    plt.title('Error change')
    plt.ylabel('Error')
    plt.xlabel('Time, sec')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(x, left, x, right)
    plt.title('Velocity change (blue - left, orange - right)')
    plt.ylabel('Velocity')
    plt.xlabel('Time, sec')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print('Done')
