from typing import List

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.qcar import PhysicalCar, VirtualOptitrack
from core.qcar import QCAR_ACTOR_ID

class StateMessageBus: # get all the state information of the car agents
    def __init__(
        self,
        qlabs: QuanserInteractiveLabs,
        agent_list: list,
        dt: float
    ) -> None:
        self.qlabs: QuanserInteractiveLabs = qlabs
        self.clinets: List[VirtualOptitrack] = [
            VirtualOptitrack(QCAR_ACTOR_ID, id, dt) for id in range(len(agent_list))
        ]

    def reset(self) -> List[np.ndarray]:
        agent_states: List[np.ndarray] = self.step()
        for state in agent_states:
            state[-3:] = 0 # initial velocity and acceleration are always 0
        return agent_states

    def step(self) -> List[np.ndarray]:
        agent_states: List[np.ndarray] = []
        for client in self.clinets:
            client.read_state(self.qlabs)
            agent_states.append(client.state)
        return agent_states
    