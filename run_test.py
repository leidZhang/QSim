# from tests.unit_test.test_dispatcher import test_dispatcher_1
from Reinformer.test import test_reinformer_car
from Reinformer.utils import test_reinformer_util
from Reinformer.main import run_reinformer_trainer
# from tests.unit_test.test_roadmap import test_generate_path_1
# from tests.pure_pursuit.test import test_purepursuite_car
# from tests.pure_pursuit.test import test_mock_optitrack_client
# from tests.pure_pursuit.test import test_dispatch_task_to_car
# from tests.replay.test import test_replay_car
from tests.client_server.test import test_client_server


if __name__ == "__main__":
    # test_dispatch_task_to_car()
    # test_replay_car()
    # test_reinformer_util()
    run_reinformer_trainer()
    # test_client_server()

