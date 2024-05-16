from core.environment import AnomalousEpisodeException
from core.policies.keyboard import KeyboardPolicy
from core.environment.primary import QLabEnvironment

from .override_demo import prepare_map_info

def limit_action(action, limit):
    if action >= limit:
        action = limit
    elif action <= -limit:
        action = -limit
    return action

def run_reward_func_demo():
    init_pos, waypoints = prepare_map_info()
    simulator: QLabEnvironment = QLabEnvironment(dt=0.05, privileged=True)
    simulator.setup(initial_state=[init_pos[0], init_pos[1], init_pos[2]], sequence=waypoints)
    controller: KeyboardPolicy = KeyboardPolicy(slow_to_zero=False)

    num_episodes = 10000
    episode_reward = 0
    for episode in range(1, num_episodes+1):
        flag: bool = True
        observation, reward, done, info = simulator.reset()
        # print(done)
        while not done:
            try:
                action = controller.execute()
                # convert to qcar actions
                action[0] = limit_action(action[0], 0.13)
                action[1] = limit_action(action[1], 0.5)
                # print(f"Action: {action}")
                observation, reward, done, info = simulator.step(action, metrics=None)
                episode_reward += reward
            except AnomalousEpisodeException as e:
                flag = False
                print(e)
                break

        # print(flag)
        if flag:
            print(f"\nEpisode {episode} completed with reward: {episode_reward}")
            episode_reward = 0
    print(f"Total reward: {episode_reward}")
