import signal
import sys
import time
import torch
from DQN import DQNAgent
from utils import get_state, perform_action, compute_reward, client
from colorama import Fore, Style
import math
import numpy as np
import airsim

num_episodes = 1000
state_size = 13  # 10 (drone state) + 3 (ball state)
action_size = 4  # [vx, vy, vz, yaw_rate]
MAX_EPISODE_DURATION = 300
MAX_STEPS = 2000
TARGET_REWARD = 1000
client = airsim.MultirotorClient()
agent = DQNAgent(state_size, action_size)

def save_progress():
    print("Saving progress...")
    torch.save(agent.q_network.state_dict(), "q_network.pth")
    torch.save(agent.target_network.state_dict(), "target_network.pth")
    with open("replay_memory.pt", "wb") as f:
        torch.save(agent.memory, f)
    print("Progress saved. Exiting.")

def signal_handler(sig, frame):
    save_progress()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    # Load the saved state dicts with weights_only set to True
    agent.q_network.load_state_dict(torch.load("q_network.pth", weights_only=True))
    agent.target_network.load_state_dict(torch.load("target_network.pth", weights_only=True))
    
    # Load the replay memory
    agent.memory = torch.load(open("replay_memory.pt", "rb"))
    print("Loaded saved models and replay memory.")
except FileNotFoundError:
    print("No saved models found. Starting fresh.")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("Reinitializing models...")
    agent = DQNAgent(state_size, action_size)  # Reinitialize if there is a loading error

def should_terminate_episode(start_time, step, total_reward, collision_info):
    current_time = time.time()
    episode_duration = current_time - start_time
    
    if collision_info.has_collided:
        print("Episode terminated due to collision.")
        return True
    
    if episode_duration >= MAX_EPISODE_DURATION:
        print(f"Episode terminated: reached max duration of {MAX_EPISODE_DURATION} seconds.")
        return True
    
    if step >= MAX_STEPS:
        print(f"Episode terminated: reached max steps of {MAX_STEPS}.")
        return True
    
    if total_reward >= TARGET_REWARD:
        print(f"Episode terminated: reached target reward of {TARGET_REWARD}.")
        return True
    
    return False

for episode in range(num_episodes):
    print(f"{Fore.CYAN}Episode {episode + 1}/{num_episodes}{Style.RESET_ALL}")

    client.reset()
    client.enableApiControl(True)
    client.takeoffAsync().join()
    
    state = get_state()
    print(f"Initial state: {state}")
    
    done = False
    total_reward = 0
    step = 0
    start_time = time.time()

    angle = 0
    radius = 200 * 0.3048  # Convert radius to meters
    altitude = -5  # Desired altitude in meters

    while not done:
        action = agent.act(state)
        print(f"Step {step + 1}, Action: {action}")

        ball_state = state[-4:]  # Assuming last 4 elements are [sphere_x, sphere_y, sphere_z, size]

        if ball_state is not None and ball_state[0] != 0 and ball_state[1] != 0:
            target_x = ball_state[0] + radius * math.cos(angle)
            target_y = ball_state[1] + radius * math.sin(angle)

            vx = target_x - state[0]  
            vy = target_y - state[1]  
            vz = altitude - state[2]   

            magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
            if magnitude > 1:
                vx, vy, vz = vx / magnitude, vy / magnitude, vz / magnitude

            action = np.array([vx, vy, vz, 0])  # Set yaw rate to 0 to focus on moving around the sphere

        # Perform the action
        perform_action(action)
        client.simSetTraceLine([0.0, 0.0, 1.0, 0.8], 10)
        # Get the next state
        next_state = get_state()
        print(f"Next state: {next_state}")

        reward = compute_reward(ball_state[:3], state[:3], action)  # Calculate reward based on the new state

        collision_info = client.simGetCollisionInfo()
        done = should_terminate_episode(start_time, step, total_reward, collision_info)

        agent.remember(state, action, reward, next_state, done)

        # Only call replay if we have enough memory
        if len(agent.memory) > agent.batch_size:
            agent.replay()

        state = next_state
        total_reward += reward
        step += 1

        angle += 0.1  # Adjust this value to control the circling speed
        if angle >= 2 * math.pi:
            angle = 0  

        if step % 10 == 0:
            print(f"Step: {step}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

    episode_duration = time.time() - start_time
    print(f"{Fore.GREEN}Episode {episode + 1} finished with total reward: {total_reward:.2f}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Episode duration: {episode_duration:.2f} seconds{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Step count: {step}{Style.RESET_ALL}")

    if (episode + 1) % 10 == 0:
        agent.update_target_network()

save_progress()