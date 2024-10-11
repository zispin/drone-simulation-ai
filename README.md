messing around with AI for drone simulation (learning project)
# Deep Q-Network (DQN) Agent for AirSim

## Overview
This project implements a Deep Q-Network (DQN) agent in AirSim, a simulation environment for drones and other vehicles. The agent is designed to track a ball and maintain a certain distance from it while keeping it in view.

## Features
- DQN implementation with target network, experience replay, and epsilon-greedy exploration
- AirSim integration for drone control and state retrieval
- Object detection using a camera and filtering out objects that don't start with "Sphere"
- Simple neural network architecture for state feature extraction

## Requirements
- Python 3.7 or later
- AirSim API (download from [AirSim GitHub](https://github.com/microsoft/AirSim))
- NumPy
- PyTorch

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/airsim-dqn.git
2. Install required pacakges:
   pip install -r requirements.txt
3. Download and install AirSim API
4. Run the script python trian.py