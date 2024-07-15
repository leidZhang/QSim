import os

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from qvl.basic_shape import QLabsBasicShape
from demos.performance_environment import prepare_test_environment, destroy_map
from core.utils.io_utils import read_npz_file
from core.policies.pure_persuit import PurePursuiteAdaptor
from .vehicle import ReplayCar


def test_replay_car() -> None:
    destroy_map() # destroy all spawned actors

    project_path: str = os.getcwd()
    folder_path: str = os.path.join(project_path, "test_data", "npzs")
    file_path: str = os.path.join(folder_path, "20240623140317447233.npz")

    replay_data: dict = read_npz_file(file_path)
    start_node_index = replay_data["task"][0][0]
    print(f"Replay task: {replay_data['task'][0]}")
    waypoints: np.ndarray = prepare_test_environment(node_id=start_node_index)

    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open('localhost')
    for i in range(0, waypoints.shape[0], 5): 
        location = [waypoints[i, 0], waypoints[i, 1], 0] 
        QLabsBasicShape(qlabs).spawn_id(
            i, location=location, 
            rotation=[0, 0, 0], 
            scale=[0.01, 0.01, 0.02], 
            configuration=1, 
            waitForConfirmation=False
        )

    car = ReplayCar(
        throttle_coeff=0.08,
        replay_data=replay_data, 
        policy_for_replay=PurePursuiteAdaptor()
    )

    task_length: int = len(replay_data["waypoints"])
    print(f"Task length: {task_length}")
    for i in range(task_length):
        car.execute(i)
    car.halt_car()
    print("Replay test completed!")

