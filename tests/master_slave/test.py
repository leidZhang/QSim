from tests.performance_environment import prepare_test_environment, destroy_map
from .coordinator import QCarCoordinator

def test_master_slave() -> None:
    destroy_map()
    prepare_test_environment(node_id=24)
    coordinator: QCarCoordinator = QCarCoordinator(duration=40)
    coordinator.start_main_process()