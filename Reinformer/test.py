import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from tests.performance_environment import destroy_map
from tests.performance_environment import prepare_test_environment
from core.policies.pt_policy import PTPolicy
from .dataset import D4RLTrajectoryDataset
from .vehicle import ReinformerCar, ReinformerPolicy
from .reinformer import ReinFormer
from .settings import *


def test_reinformer_car() -> None:
    # prepare environment
    destroy_map()
    waypoints: np.ndarray = prepare_test_environment(node_id=4)
    # nn module
    model: ReinFormer = ReinFormer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        n_blocks=N_BLOCKS,
        h_dim=EMBED_DIM,
        context_len=CONTEXT_LEN,
        n_heads=N_HEADS,
        drop_p=DROPOUT_P,
        init_temperature=INIT_TEMPERATURE,
        target_entropy=-ACT_DIM,
    ).to(DEVICE)
    # prepare policy
    max_steps: int = 100000
    state_mean: np.ndarray = np.array([
        2.21901288, 2.27527959, 1.56569062, 1.78819954, 
        0.63561087, 0.30543765, 2.1585068, 2.2505649,
        2.13603409, 2.61963549
    ])
    state_std: np.ndarray = np.array([
        1.26958866e-01, 1.08842578e+00, 2.22069285e-01, 1.02672571e+01,
        5.99335785e+00, 4.90238012e+02, 3.97721646e-01, 1.11865510e+00,
        3.78844268e-01, 1.12322966e+00
    ])
    policy: PTPolicy = ReinformerPolicy(model, model_path=MODEL_PATH)
    policy.setup(
        eval_batch_size=1,
        max_test_ep_len=max_steps,
        context_len=CONTEXT_LEN,
        state_mean=state_mean, # how to get this when deploy on a real car?
        state_std=state_std, # how to get this when deploy on a real car?
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        device=DEVICE,
    )
    # prepare car
    counter: int = 0
    actor_id: int = 0
    dt: float = 0.03
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open('localhost')
    car: ReinformerCar = ReinformerCar(
        actor_id=actor_id,
        dt=dt,
        qlabs=qlabs,
        throttle_coeff=0.08,
        steering_coeff=0.5,
    )
    car.setup(waypoints=waypoints, init_waypoint_index=0, policy=policy)
    # start the simulation
    while counter <= max_steps:
        car.execute()
        counter += 1

    