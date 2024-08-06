import struct

import numpy as np

from qvl.actor import QLabsActor
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qlabs import CommModularContainer


class Monitor:
    def __init__(self, class_id: int, actor_number: int, dt: float = 0.05) -> None:
        self.dt = dt
        self.class_id: int = class_id
        self.actor_number: int = actor_number
        self.state: np.array = np.zeros(6)
        self.last_state: np.array = np.zeros(6)

    def get_position(self, qlabs: QuanserInteractiveLabs) -> np.array:
        c: CommModularContainer = CommModularContainer()
        c.classID = self.class_id
        c.actorNumber = self.actor_number
        c.actorFunction = QLabsActor.FCN_REQUEST_WORLD_TRANSFORM
        c.payload = bytearray()
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload)

        if not qlabs.send_container(c):
            raise Exception("Failed to send container")
        c = qlabs.wait_for_container(self.class_id, self.actor_number, QLabsActor.FCN_RESPONSE_WORLD_TRANSFORM)
        x, y, _, _, _, yaw, _, _, _, = struct.unpack(">fffffffff", c.payload[0:36])
        self.state = np.array([x, y, yaw, 0, 0, 0])

    def cal_motion(self) -> None:
        x: float = self.state[0]
        y: float = self.state[1]
        yaw: float = self.state[2]
        # print(f'yaw: {yaw}')

        # calc velocity
        vx = (x - self.last_state[0]) / self.dt
        vy = (y - self.last_state[1]) / self.dt
        v = np.hypot(vx, vy)
        # calc rate of turn
        w = (yaw - self.last_state[2]) / self.dt
        # calc acceleration
        a = (v - self.last_state[3]) / self.dt
        self.state = np.array([x, y, yaw, v, w, a])

    def read_state(self, qlabs: QuanserInteractiveLabs) -> None:
        self.get_position(qlabs)
        self.cal_motion()
        self.last_state = self.state

    get_state = read_state
