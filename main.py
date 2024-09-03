import time
from typing import List
from threading import Thread, Event
from multiprocessing import Queue

import numpy as np

# from hitl import run_hitl
# from env import env
# from ego_state import relay, relay_thread
from reinformer.utils import run_reinformer_util
from reinformer.main import run_reinformer_main
from system import app
from system.routes import setup_routes
from system.settings import IP, ENV_PORT


def main() -> None:
    setup_routes(app)
    print("Starting server...")
    app.run(debug=False, host=IP, port=ENV_PORT)

    # print("Starting relay...")
    # relay_thread.start()
    # env.reset(relay)
    # while True:
    #     action = np.zeros(2)
    #     _, reward, _ = env.step(action, relay)
    #     print(f"Reward: {reward}")


if __name__ == "__main__":
    main()
    # recover_data()
    #run_reinformer_main()
