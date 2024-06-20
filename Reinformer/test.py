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
        0.31054742, 1.86853108, 0.21911925, 1.0992513, -0.05376846, 0.02852992,
        0.31109767, 1.86911335, 0.29368724, 1.88617678
    ])
    state_std: np.ndarray = np.array([
        1.66867512, 2.05809646, 1.86923593, 11.89393242, 12.85943864,
        562.16986934, 1.67460848, 2.06462373, 1.69343701, 2.04808915
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

    