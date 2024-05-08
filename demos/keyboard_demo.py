import os
from core.policies.keyboard import KeyboardController

def run_keyboard():
    policy: KeyboardController = KeyboardController()
    while True:
        policy.execute()
        print(policy.state)
        os.system("cls")