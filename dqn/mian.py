import torch
from agent import Agent
from generator.environment import CrossRoadEnvironment

def train():
    env = CrossRoadEnvironment()
    agent = Agent(action_size=5)  # Assuming 5 discrete actions
    num_episodes = 1000
    max_steps = 1000

    for episode in range(num_episodes):
        observation, reward, done, info = env.reset()
        state = agent.process_observation(observation)
        hidden = None  # Initialize LSTM hidden state
        total_reward = 0

        for t in range(max_steps):
            action, hidden = agent.select_action(state, hidden)
            next_observation, reward, done, info = env.step(action)
            next_state = agent.process_observation(next_observation)
            total_reward += reward

            # Store transition in memory
            agent.memory.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Perform optimization
            agent.optimize_model()

            if done:
                print(f"Episode {episode}, Total Reward: {total_reward}")
                break

        # Update the target network
        if episode % agent.target_update == 0:
            agent.update_target_network()

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'd3qn_per_lstm_model.pth')

if __name__ == "__main__":
    train()
