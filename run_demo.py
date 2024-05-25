import cv2

from demos.simulator_demo import simulator_demo
from demos.pid_demo.throttle_pid_demo import throttle_pid_demo
from demos.pid_demo.steering_pid_demo import steering_pid_demo
from demos.pid_demo.close_loop_demo import vision_pid_demo

if __name__ == "__main__":
    vision_pid_demo()