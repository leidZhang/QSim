from multiprocessing import Process

import numpy as np

from pal.products.qcar import QCar
from core.qcar import QCAR_ACTOR_ID
from core.qcar.virtual import VirtualRuningGear
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from core.environment.simulator import QLabSimulator


if __name__ == "__main__":
    sim = QLabSimulator(offsets=(0, 0))
    print("Rendering map...")
    sim.render_map()
    print("Resetting car pos...")
    sim.reset_map([0, 0, 0], [0, 0, np.pi/2])
    print("Adding new car...")
    sim.add_car([0, 2, 0], [0, 0, np.pi/2])

    car1 = QCar(id=0)
 
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")
    virtual_car = VirtualRuningGear(QCAR_ACTOR_ID, 1)
    while True:
        virtual_car.read_write_std(qlabs, 0.08, 0.1)
        car1.read_write_std(0.08, 0.1)