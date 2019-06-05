from __future__ import print_function
from lib.env import VrepEnvironment
from lib.agents import Pioneer, Display
import settings, time, argparse
import matplotlib.pyplot as plt
import numpy as np
import keyboard


def loop(robot):
    """1"""


if __name__ == "__main__":
    isPressedkey = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help='Test in a room environment')
    args = parser.parse_args()

    env = args.test

    if env:
        environment = VrepEnvironment(
            settings.SCENES + '/2.ttt')  # Open the file containing our scene (robot and its environment)
    else:
        environment = VrepEnvironment(
            settings.SCENES + '/2.ttt')  # Open the file containing our scene (robot and its environment)
    environment.start_vrep()
    environment.connect()

    robot = Pioneer(environment)
    display = Display(robot, True)

    try:
        # LOOP
        start = time.time()
        step = 0
        environment.start_simulation()
        environment.step_simulation()
        camera_handle = environment.get_handle('Vision_sensor')

        while step < settings.simulation_steps:

            if not keyboard.is_pressed('w') and not keyboard.is_pressed('s') and not keyboard.is_pressed(
                    'a') and not keyboard.is_pressed('d'):
                robot.change_velocity([0, 0])
                speed = 0

            if keyboard.is_pressed('w'):
                speed = 0.5
                print('\n\nw')
                robot.change_velocity([speed, speed])
                isPressedkey = 1
            else:
                if not isPressedkey:
                    speed = 0
                    robot.change_velocity([speed, speed])
                    isPressedkey = 0

            if keyboard.is_pressed('a'):
                robot.change_velocity([speed - 0.2, speed + 0.2])
                isPressedkey = 1
            else:
                if not isPressedkey:
                    robot.change_velocity([0, 0])
                    isPressedkey = 0

            if keyboard.is_pressed('d'):
                robot.change_velocity([speed + 0.2, speed - 0.2])
                isPressedkey = 1
            else:
                if not isPressedkey:
                    robot.change_velocity([0, 0])
                    isPressedkey = 0

            if keyboard.is_pressed('s'):
                speed = -0.5
                robot.change_velocity([speed, speed])
                isPressedkey = 1
            else:
                if not isPressedkey:
                    speed = 0
                    robot.change_velocity([0, 0])
                    isPressedkey = 0

            display.update()
            environment.step_simulation()  # Advance the simulation by one step
            loop(robot)
            step += 1
            if robot.pos[0] < 330:
                end = time.time()
                print('Track completed!\nTime: {}'.format(end - start))
                environment.destroy_instances()
                break
            if robot.distance_closest()[0] < 0.25 or robot.distance_closest()[1] < .25:
                print('Hit an obstacle!\nTime: {}'.format(time.time() - start))
                environment.destroy_instances()
                break

    except KeyboardInterrupt:
        end = time.time()
        print('\n\nInterrupted! Time: {}s'.format(end - start))
        environment.destroy_instances()
