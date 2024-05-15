import cv2

from demos.keyboard_demo import run_keyboard
# from demos.override_demo import run_override_demo
from demos.pure_persuite.demo import start_test_mp
from demos.reward_func_demo import run_reward_func_demo
from core.sensor.sensor import VirtualCSICamera

if __name__ == "__main__":
    start_test_mp()