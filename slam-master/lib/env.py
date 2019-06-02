import time, subprocess, os
import vrep
import settings

class VrepEnvironment(object):

    def __init__(self, path):

        self.path           = path
        self.address        = settings.local_ip # Local IP
        self.port           = settings.sim_port # Port for Remote API
        self.frames_elapsed = 0
        self.max_attempts   = 2
        self.client_id      = None
        self.connected      = False
        self.running        = False
        self.scene_loaded   = False
        self.headless       = False
        self.start_stop_delay = .3


        self.modes = {'blocking'  : vrep.simx_opmode_blocking,  # Waits until the respose from vrep remote API is sent back
                      'oneshot'   : vrep.simx_opmode_oneshot,   # Sends a one time packet to vrep remote API (used for setting parameters)
                      'streaming' : vrep.simx_opmode_streaming, # Sends a signal to start sending packets every frame of the simulation back to the python client
                      'buffer'    : vrep.simx_opmode_buffer}    # To be used in combination with 'streaming' to obtain the information being streamed

    def start_vrep(self, exit_after_sim=False, headless=False, verbose=False):
        """
        Launches a new V-REP instance using subprocess
        If you want to connect to an existing instance, use self.connect() with a correct port instead
            arguments:
                [sim_port]      : VREP simulation port, set by '-gREMOTEAPISERVERSERVICE_port_TRUE_TRUE'
                [path]          : path of the V-REP scene (.ttt format) to run
                [exit_ater_sim] : corresponds to command line argument '-q' (exits after simulation)
                [headless]      : corresponds to '-h' (runs without a GUI)
                [verbose]       : suppress prompt messages
        """

        command = 'bash ' + settings.VREP_DIR + '/vrep.sh'  + \
                ' -gREMOTEAPISERVERSERVICE_{}_FALSE_TRUE'.format(settings.sim_port) # Start remote API at a specified port

        # Additional arguments
        if headless:
            command += ' -h'
        if exit_after_sim:
            command += ' -q'

        # command += ' -s'                     # Start sim
        command += ' ' + self.path                # Scene name
        command += ' &'                      # Non-blocking call

        # Call the process and start VREP
        print "Launching V-REP at port {}".format(settings.sim_port)
        DEVNULL = open(os.devnull, 'wb')
        try:
            if verbose:
                subprocess.Popen(command.split())
            else:
                subprocess.Popen(command.split(), stdout=DEVNULL, stderr=DEVNULL)
        except:
            print "Please set up the correct path to V-REP in '../nao_rl/settings.py' (VREP_DIR = '/path/to/VREP_FOLDER_NAME')"
        time.sleep(1)

    @staticmethod
    def destroy_instances():
        print "Destroying all previous VREP instances..."
        subprocess.Popen('pkill vrep'.split())
        time.sleep(1)

    def connect(self):
        if self.connected:
            raise RuntimeError('Client is already connected.')
        e = 0
        c_id = 0
        while e < self.max_attempts:
            c_id = vrep.simxStart(self.address, self.port, True, True, 5000, 0)
            if c_id >= 0:
                self.client_id = c_id
                self.connected = True
                print 'Connection to client successful. IP: {}, port: {}, client id: {}'.format(self.address, self.port, c_id)
                break
            else:
                e += 1
                print 'Could not connect to client, attempt {}/{}...'.format(e, self.max_attempts)
                time.sleep(1)
        self.set_boolean_parameter(0, False)
        self.set_boolean_parameter(12, False)

    def disconnect(self):
        if not self.connected:
            raise RuntimeError('Client is not connected.')
        vrep.simxFinish(self.client_id)
        self.connected = False

    def load_scene(self, path):
        if self.scene_loaded:
            raise RuntimeError('Scene is loaded already.')
        vrep.simxLoadScene(self.client_id, path, 0, self.modes['blocking'])
        self.scene_loaded = True

    def close_scene(self):
        if not self.scene_loaded:
            raise RuntimeError('Scene is not loaded')
        vrep.simxCloseScene(self.client_id, self.modes['blocking'])
        self.scene_loaded = False

    def start_simulation(self):
        vrep.simxSynchronous(self.client_id, True)

        if self.headless:
            self.set_boolean_parameter(vrep.sim_boolparam_threaded_rendering_enabled, True)

        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)
        self.running = True
        time.sleep(self.start_stop_delay)

    def stop_simulation(self):
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_blocking)
        self.running = False
        time.sleep(self.start_stop_delay)

    def step_simulation(self):
        vrep.simxSynchronousTrigger(self.client_id)
        self.frames_elapsed += 1

    def set_boolean_parameter(self, parameter_id, value):
        vrep.simxSetBooleanParameter(self.client_id, parameter_id, value, vrep.simx_opmode_blocking)

    def get_handle(self, name):
        return vrep.simxGetObjectHandle(self.client_id, name, vrep.simx_opmode_blocking)[1]

    def get_joint_angle(self, handle, mode='blocking'):
        return vrep.simxGetJointPosition(self.client_id, handle, self.modes[mode])[1]

    def get_vision_image(self, handle, mode='blocking'):
        res, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, handle, 0, self.modes[mode])
        return res, resolution, image

    def read_lidar(self, handle, mode='blocking'):
        info = vrep.simxReadVisionSensor(self.client_id, handle, self.modes[mode])
        return info

    def set_target_velocity(self, handle, vel):
        vrep.simxSetJointTargetVelocity(self.client_id, handle, vel, vrep.simx_opmode_oneshot)
