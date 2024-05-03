from qvl.actor import QLabsActor
import math



######################### MODULAR CONTAINER CLASS #########################

class QLabsTrafficCone(QLabsActor):
    """This class is for spawning traffic cones."""

    ID_TRAFFIC_CONE = 10000
    """Class ID"""

    def __init__(self, qlabs, verbose=False):
       """ Constructor Method

       :param qlabs: A QuanserInteractiveLabs object
       :param verbose: (Optional) Print error information to the console.
       :type qlabs: object
       :type verbose: boolean
       """

       self._qlabs = qlabs
       self._verbose = verbose
       self.classID = self.ID_TRAFFIC_CONE
       return
