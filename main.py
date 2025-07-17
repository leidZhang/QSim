# from tests.unit_test.test_dispatcher import test_dispatcher_1
# from tests.unit_test.test_roadmap import test_generate_path_1
from demos.pure_pursuit.test import test_purepursuite_car
from demos.pure_pursuit.test import test_mock_optitrack_client
from demos.pure_pursuit.test import test_dispatch_task_to_car
from demos.pure_pursuit_2.test import test_dispatch_task_to_car_2
from demos.replay.test import test_replay_car

if __name__ == "__main__":
    # test_dispatch_task_to_car()
    # test_replay_car()
    # test_reinformer_util()
    test_dispatch_task_to_car_2()
