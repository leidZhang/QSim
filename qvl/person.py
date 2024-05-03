from qvl.qlabs import CommModularContainer
from qvl.character import QLabsCharacter
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsPerson(QLabsCharacter):
    """ This class implements spawning and AI navigation of the environment for human pedestrians."""

    ID_PERSON = 10030

    STANDING = 0
    """ Speed constant for the move_to method. """
    WALK = 1.2
    """ Speed constant for the move_to method. """
    JOG = 3.6
    """ Speed constant for the move_to method. """
    RUN = 6.0
    """ Speed constant for the move_to method. """


    def __init__(self, qlabs, verbose=False):
       """ Constructor method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_PERSON
       return


