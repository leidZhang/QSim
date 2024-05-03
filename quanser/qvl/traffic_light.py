from qvl.qlabs import CommModularContainer
from qvl.actor import QLabsActor

import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsTrafficLight(QLabsActor):


    ID_TRAFFIC_LIGHT = 10051
    """Class ID"""

    FCN_TRAFFIC_LIGHT_SET_STATE = 10
    FCN_TRAFFIC_LIGHT_SET_STATE_ACK = 11

    STATE_RED = 0
    """State constant for red light"""
    STATE_GREEN = 1
    """State constant for green light"""
    STATE_YELLOW = 2
    """State constant for yellow light"""

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_TRAFFIC_LIGHT
       return

    def set_state(self, state, waitForConfirmation=True):
        """Set the light state (red/yellow/green) of a traffic light actor

        :param state: An integer constant corresponding to a light state (see class constants)
        :param waitForConfirmation: (Optional) Wait for confirmation of the state change before proceeding. This makes the method a blocking operation.
        :type state: uint32
        :type waitForConfirmation: boolean
        :return: `True` if successful, `False` otherwise
        :rtype: boolean

        """

        if (not self._is_actor_number_valid()):
            return False

        c = CommModularContainer()
        c.classID = self.ID_TRAFFIC_LIGHT
        c.actorNumber = self.actorNumber
        c.actorFunction = self.FCN_TRAFFIC_LIGHT_SET_STATE
        c.payload = bytearray(struct.pack(">B", state))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if waitForConfirmation:
            self._qlabs.flush_receive()

        if (self._qlabs.send_container(c)):
            if waitForConfirmation:
                c = self._qlabs.wait_for_container(self.ID_TRAFFIC_LIGHT, self.actorNumber, self.FCN_TRAFFIC_LIGHT_SET_STATE_ACK)
                if (c == None):
                    return False

            return True
        else:
            return False

