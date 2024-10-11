import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-4, batch_size=64, max_memory_size=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory_size)
        self.priorities = deque(maxlen=max_memory_size)  # For storing priorities
        self.use_double_dqn = True
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.update_target_network()

    def remember(self, state, action, reward, next_state, done):
        # Assign a priority to the experience (initially set to 1.0)
        priority = 1.0
        self.memory.append(Experience(state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample_experience(self):
        if len(self.memory) < self.batch_size:
            return [], []  # or raise an error, or return a smaller batch
        print(f"Memory size: {len(self.memory)}, Batch size: {self.batch_size}")
        probabilities = self.get_probabilities()  # Assuming this method gets the probabilities
        print(f"Probabilities shape: {probabilities.shape}, Memory shape: {len(self.memory)}")
        indices = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities)
        experiences = [self.memory[i] for i in indices]
        return experiences, indices


    def update_priorities(self, indices, errors):
        # Update the priorities based on TD errors
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Add a small constant to prevent zero priority

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_probabilities(self):
        # Assuming you're calculating priorities or some form of probabilities
        priorities = np.array([experience.priority for experience in self.memory])  # example
        probabilities = priorities / np.sum(priorities)  # Normalize to get a valid probability distribution
        return probabilities


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Select a random action from the action space (uniform distribution)
            return np.random.uniform(-1, 1, self.action_size)  # Random continuous action
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.q_network(state)

        return action_values.cpu().data.numpy()[0]  # Ensure it returns an array

    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample experiences based on priorities
        experiences, indices = self.sample_experience()
        if experiences is None:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*experiences)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
        action_batch = torch.LongTensor(np.array(action_batch)).to(device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(device)

        # Get current Q values
        current_q_values = self.q_network(state_batch)

        # Double DQN logic
        with torch.no_grad():
            next_action = self.q_network(next_state_batch).max(1)[1]  # Get actions from Q-network
            target_q_values = self.target_network(next_state_batch).gather(1, next_action.unsqueeze(1)).squeeze(1)

        # Calculate target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * target_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities based on TD error
        errors = torch.abs(current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1) - target_q_values).cpu().numpy()
        self.update_priorities(indices, errors)
