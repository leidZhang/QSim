from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsQubeServo2:

    ID_QUBE_SERVO_2 = 60


    # Initilize class
    def __init__(self):

       return


    def spawn(self, qlabs, deviceNumber, location, rotation, configuration=0, waitForConfirmation=True):
        return qlabs.spawn(deviceNumber, self.ID_QUBE_SERVO_2, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, configuration, waitForConfirmation)

    def spawn_degrees(self, qlabs, deviceNumber, location, rotation, configuration=0, waitForConfirmation=True):

        return qlabs.spawn(deviceNumber, self.ID_QUBE_SERVO_2, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, configuration, waitForConfirmation)

