from qvl.qlabs import CommModularContainer
from qvl.character import QLabsCharacter
import math

import struct


######################### MODULAR CONTAINER CLASS #########################

class QLabsAnimal(QLabsCharacter):
    """ This class implements spawning and AI navigation of the environment for animals."""

    ID_ANIMAL = 10031

    GOAT = 0
    """ Configuration constant. """
    SHEEP = 1
    """ Configuration constant. """
    COW = 2
    """ Configuration constant. """


    GOAT_STANDING = 0
    """ Speed constant for the move_to method. """
    GOAT_WALK = 0.8
    """ Speed constant for the move_to method. """
    GOAT_RUN = 4.0
    """ Speed constant for the move_to method. """

    SHEEP_STANDING = 0
    """ Speed constant for the move_to method. """
    SHEEP_WALK = 0.60
    """ Speed constant for the move_to method. """
    SHEEP_RUN = 3.0
    """ Speed constant for the move_to method. """

    COW_STANDING = 0
    """ Speed constant for the move_to method. """
    COW_WALK = 1.0
    """ Speed constant for the move_to method. """
    COW_RUN = 6.0
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
       self.classID = self.ID_ANIMAL
       return


