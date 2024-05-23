import cv2

from demos.simulator_demo import simulator_demo
from demos.close_loop.throttle_pid_demo import throttle_pid_demo
from demos.close_loop.steering_pid_demo import steering_pid_demo

if __name__ == "__main__":
    steering_pid_demo()