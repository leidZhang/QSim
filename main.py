from core.utils.executions import BaseCoordinator
from demos.performance_environment import destroy_map
from demos.slam_purepursuit.coordinator import LidarSLAMCoordinator


# 11, 0, 2, 4, 14, 16, 18, 11
# 10, 2, 4, 14, 20, 22, 10
def run_demo() -> None:
    destroy_map() # destroy the map
    coordinator: BaseCoordinator = LidarSLAMCoordinator(queue_size=5)
    coordinator.start()


if __name__ == "__main__":
    run_demo()