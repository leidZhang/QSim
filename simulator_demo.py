from qvl.qlabs import QuanserInteractiveLabs 
from qvl.basic_shape import QLabsBasicShape
from core.policies.pure_persuit import PurePursuitPolicy
from core.environment import QLabEnvironment
from core.roadmap import ACCRoadMap

if __name__ == "__main__":
    roadmap: ACCRoadMap = ACCRoadMap()
    qlabs = QuanserInteractiveLabs()
    node_id: int = 24
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    waypoint_sequence = roadmap.generate_path([10, 4, 14, 20, 22, 10])

    simulator: QLabEnvironment = QLabEnvironment(dt=0.05, privileged=True)
    simulator.setup(initial_state=[x_pos, y_pose, angle], sequence=waypoint_sequence)
    policy: PurePursuitPolicy = PurePursuitPolicy(max_lookahead_distance=0.75)

    num_episodes = 10000
    for episode in range(1, num_episodes+1):
        observation, reward, done, info = simulator.reset()
        while not done: 
            action, metrics = policy(observation)
            observation, reward, done, info = simulator.step(action, metrics)
            print(f"action: {action}, step_reward: {reward}, done: {done}")
        print(f"Episode {episode} completed")