import numpy as np

from core.roadmap import ACCRoadMap
from core.environment import QLabEnvironment
from core.policies.pure_persuit import PurePursuitPolicy
from core.policies.keyboard import KeyboardPolicy

def prepare_map_info(node_id: int = 24) -> tuple:
    roadmap: ACCRoadMap = ACCRoadMap()
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    waypoint_sequence = roadmap.generate_path([10, 4, 14, 20, 22, 10])
    return [x_pos, y_pose, angle], waypoint_sequence

def limit_action(agent_action: float, human_action: float, limit: float):
    action: float = agent_action + human_action
    if action >= limit:
        action = limit
    elif action <= -limit:
        action = -limit
    return action

def run_override_demo(): # simple human in the loop
    init_pos, waypoints = prepare_map_info()
    simulator: QLabEnvironment = QLabEnvironment(dt=0.05, privileged=True)
    simulator.setup(initial_state=[init_pos[0], init_pos[1], init_pos[2]], sequence=waypoints)
    policy: PurePursuitPolicy = PurePursuitPolicy(max_lookahead_distance=0.75)
    controller: KeyboardPolicy = KeyboardPolicy()

    action: np.ndarray = np.zeros(2)
    num_episodes = 10000
    for episode in range(1, num_episodes+1):
        observation, reward, done, info = simulator.reset()
        while not done:
            # get agent action and human action
            agent_action, metrics = policy(observation)
            human_action = controller.execute()
            # get final action
            action[0] = limit_action(agent_action[0], human_action[0], 0.13)
            action[1] = limit_action(agent_action[1], human_action[1], 0.5)

            observation, reward, done, info = simulator.step(action, metrics)
            # print(f"agent: {agent_action}, human: {human_action}, final: {action}")
            # print(f"action: {agent_action}, step_reward: {reward}, done: {done}")
        print(f"Episode {episode} completed")