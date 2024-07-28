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


class VirtualRuningGear:
    FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE = 10
    FCN_QCAR_VELOCITY_STATE_RESPONSE = 11

    def __init__(self, class_id: int, actor_number: int) -> None:
        self.class_id: int = class_id
        self.actor_number: int = actor_number

    def read_write_std(self, qlabs: QuanserInteractiveLabs, throttle, steering, leds=None) -> None:
        c = CommModularContainer()
        c.classID = self.class_id
        c.actorNumber = self.actor_number       
        c.actorFunction = self.FCN_QCAR_SET_VELOCITY_AND_REQUEST_STATE
        c.payload = bytearray(struct.pack(">ffBBBBB", throttle * 7.5, -steering, False, False, False, False, False))
        c.containerSize = c.BASE_CONTAINER_SIZE + len(c.payload) 

        location = [0,0,0]
        rotation = [0,0,0]
        frontHit = False
        rearHit = False

        qlabs.flush_receive()

        if (qlabs.send_container(c)):
            c = qlabs.wait_for_container(self.class_id, self.actor_number, self.FCN_QCAR_VELOCITY_STATE_RESPONSE)

            if (c == None):
                print("No response from QCar")
                return False, location, rotation, frontHit, rearHit

            if len(c.payload) == 26:
                location[0], location[1], location[2], rotation[0], rotation[1], rotation[2], frontHit, rearHit, = struct.unpack(">ffffff??", c.payload[0:26])
                return True, location, rotation, frontHit, rearHit
            else:
                print("Invalid response from QCar")
                return False, location, rotation, frontHit, rearHit
        else:
            print("Failed to send container")
            return False, location, rotation, frontHit, rearHit