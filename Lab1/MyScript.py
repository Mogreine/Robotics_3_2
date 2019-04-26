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
    initial_speed = 0.2
    maintained_dist_right = 0.6
    maintained_dist_front = 1

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

        e, self.lidar_handle1 = vrep.simxGetObjectHandle(self.client_id, 'fastHokuyo',
                                                         vrep.simx_opmode_oneshot_wait)
        self.check_on_error(e, "Can not find lidar 1", True)

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

    def fix_distance(self, dist, old_dist, front_dist):
        # constants
        kp = 0.5
        kd = 1
        ki = 0.01
        front_k = 100
        iMin, iMax = -0.2, 0.2

        # error calculation
        if front_dist == 0:
            front_dist = 1e9
        self.e = self.maintained_dist_right - dist + (self.maintained_dist_front / (front_dist ** 10))

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

        prev_dist = 0
        prev_left_detection = True
        while vrep.simxGetConnectionId(self.client_id) != -1:
            front_detection, left_detection, right_detection, distance_front, distance_left, distance_right = self.get_proximity_data()
            front = 0 if not front_detection else distance_front
            e_lidar, points = self.get_lidar_data()
            print(points)
            if left_detection and right_detection:
                if not prev_left_detection:
                    self.set_motor_speed(self.initial_speed, self.initial_speed)
                else:
                    dist = self.calc_dist(distance_left, distance_right)
                    print("Distance: {}".format(dist))
                    self.fix_distance(dist, prev_dist, front)
                    prev_dist = dist
            elif not left_detection:
                print("Turning")
                self.fix_distance(0.7, prev_dist, front)
                prev_dist = 0.7
            elif left_detection:
                # self.set_motor_speed(self.initial_speed, self.initial_speed)
                self.fix_distance(0.6, prev_dist, front)
                prev_dist = 0.6
            time.sleep(0.01)
            prev_left_detection = left_detection

        vrep.simxFinish(self.client_id)

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

        return front_detection, left_detection, right_detection, np.linalg.norm(front), np.linalg.norm(left), np.linalg.norm(right)

    def get_lidar_data(self):
        e, data = vrep.simxGetStringSignal(self.client_id, "lidarMeasuredData", vrep.simx_opmode_buffer)
        self.check_on_error(e, "simxGetStringSignal lidar error")
        measured_data = vrep.simxUnpackFloats(data)
        point_data = np.reshape(measured_data, (int(len(measured_data) / 3), 3))
        return e, point_data


if __name__ == '__main__':
    robot = Robot()
    robot.start_simulation()