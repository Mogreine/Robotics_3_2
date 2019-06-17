import cv2
import math
import numpy as np


class Map(object):
    def __init__(self, width, height, meters):
        self.width = width
        self.height = height
        self.start_pos_x = width // 2
        self.start_pos_y = height // 5
        self.image = np.zeros((height, width, 3), np.uint8)
        self.line_color = (255, 255, 255)
        self.robot_color = (0, 255, 0)
        self.ignore_dist = 300
        self.scale_effect = 20

    # Метод обновляет карту.
    def update(self, pos, angle, lidar_data):
        pos = [p * self.scale_effect for p in pos]
        lidar_data = [dt * self.scale_effect for dt in lidar_data]
        pos[0] += self.start_pos_x
        pos[1] += self.start_pos_y
        pos = (int(pos[0]), int(pos[1]))
        # drawing lidar lines
        self.draw_lidar_data(pos, angle, lidar_data)

        # draw the vehicle as a circle
        self.draw_robot(pos)

        # show map
        self.show_map()

    # Показывает карту. При нажатии на кнопку "s" сохраняет карту в директорию скрипта.
    def show_map(self):
        cv2.imshow('SLAM', self.image)
        k = cv2.waitKey(1) & 0xFF
        # saving the map if 's' is pressed
        if k == ord('s'):
            cv2.imwrite('map.png', self.image)

    # Отрисовывает самого робота на карте
    def draw_robot(self, pos):
        cv2.circle(self.image, pos, 4, (0, 255, 0), cv2.FILLED)

    # Производит отрисовку данных в виде лучей, в случае если длина луча меньше максимальной (< 95) значит
    # он врезается в стену и нужно ее сделать немного жирнее, чтобы точно детектилась
    def draw_lidar_data(self, pos, ang, lidar_data):
        print("fir - {}, mid - {}, last - {}".format(lidar_data[0], lidar_data[684 // 2], lidar_data[684 - 3]))
        angle = -30 - ang
        angle_step = 240.0 / len(lidar_data)
        for dist in lidar_data:
            x = int(pos[0] + dist * math.cos(math.radians(angle)))
            y = int(pos[1] + dist * math.sin(math.radians(angle)))
            cv2.line(self.image, pos, (x, y), self.line_color, 1)
            if dist < 95:
                cv2.circle(self.image, (x, y), 5, (0, 0, 0), cv2.FILLED)
            angle += angle_step


