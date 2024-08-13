import torch
import random
import gymnasium as gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dqn_input(state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2):
    
    Cosine_of_theta1_tensor = np.digitize((state[0]), Cosine_of_theta1)
    Sine_of_theta1_tensor = np.digitize((state[1]), Sine_of_theta1)
    Cosine_of_theta2_tensor = np.digitize((state[2]), Cosine_of_theta2)
    Sine_of_theta2_tensor = np.digitize((state[3]), Sine_of_theta2)
    Angular_velocity_of_theta1_tensor = np.digitize((state[4]), Angular_velocity_of_theta1)
    Angular_velocity_of_theta2_tensor = np.digitize((state[5]), Angular_velocity_of_theta2)

    input_tensor = torch.Tensor([Cosine_of_theta1_tensor, Sine_of_theta1_tensor, Cosine_of_theta2_tensor, Sine_of_theta2_tensor, Angular_velocity_of_theta1_tensor, Angular_velocity_of_theta2_tensor]).to(device)
     
    return input_tensor

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, h3_nodes, out_actions):
        super(DQN, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.fc3 = nn.Linear(h2_nodes, h3_nodes)
        self.out = nn.Linear(h3_nodes, out_actions)

    def forward(self, x):
        # Define the forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x
    
env = gym.make('Acrobot-v1', render_mode="human")
    	
Cosine_of_theta1 = np.linspace(-1, 1, 20) # Between -1 and 1
Sine_of_theta1 = np.linspace(-1, 1, 20) # Between -1 and 1
Cosine_of_theta2 = np.linspace(-1, 1, 20) # Between -1 and 1
Sine_of_theta2 = np.linspace(-1, 1, 20) # Between -1 and 1
Angular_velocity_of_theta1 = np.linspace(-12.56, 12.56, 100) # betwen -12.56 and 12.56
Angular_velocity_of_theta2 = np.linspace(-12.56, 12.56, 100) # betwen -12.56 and 12.56

in_states = 6
h1_nodes = 32
h2_nodes = 32
h3_nodes = 32
out_actions = 3
policy_dqn = DQN(in_states, h1_nodes, h2_nodes, h3_nodes, out_actions).to(device)

# Load the trained model weights
policy_dqn.load_state_dict(torch.load("CartPole.pt"))

# Switch the model to evaluation mode
policy_dqn.eval()

totel_steps = 0
run = 0
real_run = 0
    


while True:
    now_state = env.reset()[0]  # Reset environment and get initial state
    done = False  # Flag to check if the episode is finished
    step = 0
    run += 1  # Increment the episode counte
    # Play one episode
    while not done and step < 1000:
        # Use the policy network to select the best action
        with torch.no_grad():
            action = policy_dqn(dqn_input(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)).argmax().item()  # Best action
        step += 1
        totel_steps += 1
        # Take action and observe result
        new_state, reward, done, truncated, _ = env.step(action)
        
        # Store transition in memory
        now_state = new_state
    print(step)