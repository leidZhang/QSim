#!/usr/bin/env python3

import sys
import time
from queue import Queue
from socket import socket
from typing import Dict, Union, Any

import numpy as np

from core.datatypes.pose import MockPose, Pose
from .NatNetClient import NatNetClient


class OptitrackClient:
    def __init__(self, data_queue: Queue, optitrack_object_id: int = 1) -> None:
        self.optitrack_object_id = optitrack_object_id
        self.data_queue: Queue = data_queue
        self.streaming_client: NatNetClient = NatNetClient()
        self.option_dict: Dict[str, Union[str, bool]] = {
            "clientAddress": "127.0.0.1", # local host
            "serverAddress": "192.168.2.11", # default server address
            "use_multicast": True # default multicast setting
        }

        self.t = time.time()
        self.last_pose: Pose = Pose(1, (0, 0, 0), (0, 0, 0, 0))

    def mount(self, option_dict: Dict[str, Union[str, bool]] = None) -> None:
        self.t: float = time.time()
        if option_dict is not None:
            self.option_dict = option_dict

        self.streaming_client.set_client_address(self.option_dict["clientAddress"])
        self.streaming_client.set_server_address(self.option_dict["serverAddress"])
        self.streaming_client.set_use_multicast(self.option_dict["use_multicast"])
        self.streaming_client.rigid_body_listener = self.receive_rigid_body_frame
        
        return self.streaming_client.run()

    def receive_rigid_body_frame(self, id, pos, rot):
        # print(f"ID: {id}")
        if id != self.optitrack_object_id:
            return
        
        pose: Pose = Pose(id, pos, rot)
        # duration: float = time.time() - self.t
        # v_x: float = (pose.position_x - self.last_pose.position_x) / duration
        # v_y: float = (pose.position_y - self.last_pose.position_y) / duration
        # v_w: float = (pose.angular_speed - self.last_pose.angular_speed) / duration

        # pose.set_velocity_x(v_x)
        # pose.set_velocity_y(v_y) 
        # pose.set_angualr_speed(v_w)         

        if self.data_queue.full():
            self.data_queue.get()
        self.data_queue.put(pose)

        # self.t = time.time()
        # self.last_pose = pose
        # print("DT: {}".format(time.time() - self.t))


        
