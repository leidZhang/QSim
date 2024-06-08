from tests.performance_environment import prepare_test_environment, destroy_map
from .scripts import Master
from .constants import THROTTLE_DEFAULT_K_P, THROTTLE_DEFAULT_K_D, THROTTLE_DEFAULT_K_I

def test_master_slave() -> None:
    destroy_map()
    prepare_test_environment(node_id=24)
    master: Master = Master(duration=40)
    master.start_main_process()