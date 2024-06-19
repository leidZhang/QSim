import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from tests.performance_environment import destroy_map
from tests.performance_environment import prepare_test_environment
from core.policies.pt_policy import PTPolicy
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
    policy: PTPolicy = ReinformerPolicy(model, model_path=MODEL_PATH)
    policy.setup(
        eval_batch_size=1,
        max_test_ep_len=max_steps,
        context_len=CONTEXT_LEN,
        state_mean=0.0, # how to get this when deploy on a real car?
        state_std=0.0, # how to get this when deploy on a real car?
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

    