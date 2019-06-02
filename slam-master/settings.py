import os, random

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
SCENES = MAIN_DIR + '/scenes'
VREP_DIR = MAIN_DIR + '/vrep'

sim_port = 19000
local_ip = '127.0.0.1'

fps = 30.
simulation_steps = 5000
image_size = 800
map_size   = 40
steps_slam = 10
