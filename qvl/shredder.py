from qvl.qlabs import QuanserInteractiveLabs, CommModularContainer
from quanser.common import GenericError
from qvl.actor import QLabsActor
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsShredder(QLabsActor):


    ID_SHREDDER = 190

    RED = 0
    GREEN = 1
    BLUE = 2
    WHITE = 3

    # Initialize class
    def __init__(self, qlabs, verbose=False):
        """ Constructor Method

        :param qlabs: A QuanserInteractiveLabs object
        :param verbose: (Optional) Print error information to the console.
        :type qlabs: object
        :type verbose: boolean
        """
        
        self._qlabs = qlabs
        self._verbose = verbose
        self.classID = self.ID_SHREDDER
        return


    '''
    def spawn(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):
        return qlabs.spawn(actorNumber, self.ID_SHREDDER, location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], 1, 1, 1, configuration, waitForConfirmation)

    def spawn_degrees(self, qlabs, actorNumber, location, rotation, configuration=0, waitForConfirmation=True):

        return qlabs.spawn(actorNumber, self.ID_SHREDDER, location[0], location[1], location[2], rotation[0]/180*math.pi, rotation[1]/180*math.pi, rotation[2]/180*math.pi, 1, 1, 1, configuration, waitForConfirmation)
    '''
