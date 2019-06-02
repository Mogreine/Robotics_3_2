import cv2
import math
import numpy as np


class Map(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.start_pos_x = width // 2
        self.start_pos_y = height // 2
        self.image = np.zeros((height, width, 3), np.uint8)
        self.line_color = (255, 255, 255)
        self.robot_color = (0, 255, 0)
        self.ignore_dist = 300
        # self.scale_effect =

    def update(self, pos, angle, lidar_data):
        pos[0] += self.start_pos_x
        pos[1] += self.start_pos_y
        pos = (int(pos[0]), int(pos[1]))
        # drawing lidar lines
        self.draw_lidar_data(pos, angle, lidar_data)

        # draw the vehicle as a circle
        self.draw_robot(pos)

        # show map
        self.show_map()

    def show_map(self):
        cv2.imshow('SLAM', self.image)
        cv2.waitKey(1)

    def draw_robot(self, pos):
        cv2.circle(self.image, pos, 4, (2, 2, 2), cv2.FILLED)

    def draw_lidar_data(self, pos, ang, lidar_data):
        print(ang)
        angle = -30 + ang - 44
        angle_step = 240.0 / len(lidar_data)
        print(lidar_data[0])
        for dist in lidar_data:
            x = int(pos[0] + dist * math.cos(math.radians(angle)))
            y = int(pos[1] + dist * math.sin(math.radians(angle)))
            angle += angle_step
            # print(x, y)
            cv2.line(self.image, pos, (x, y), self.line_color, 1)
