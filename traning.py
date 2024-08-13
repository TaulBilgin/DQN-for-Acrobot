import torch
import random
import gymnasium as gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change the status to dqn tensor
def dqn_input(state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2):
    Cosine_of_theta1_tensor = np.digitize((state[0]), Cosine_of_theta1)
    Sine_of_theta1_tensor = np.digitize((state[1]), Sine_of_theta1)
    Cosine_of_theta2_tensor = np.digitize((state[2]), Cosine_of_theta2)
    Sine_of_theta2_tensor = np.digitize((state[3]), Sine_of_theta2)
    Angular_velocity_of_theta1_tensor = np.digitize((state[4]), Angular_velocity_of_theta1)
    Angular_velocity_of_theta2_tensor = np.digitize((state[5]), Angular_velocity_of_theta2)

    input_tensor = torch.Tensor([Cosine_of_theta1_tensor, Sine_of_theta1_tensor, Cosine_of_theta2_tensor, Sine_of_theta2_tensor, Angular_velocity_of_theta1_tensor, Angular_velocity_of_theta2_tensor]).to(device)
     
    return input_tensor

def optimize(memory, policy_dqn, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2, learning_rate, gamma, finis):
    # Sample a batch of transitions from memory
    random_memoy = memory.sample(64)

    # Initialize the optimizer and loss function
    optimizer = torch.optim.AdamW(policy_dqn.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()

    current_q_list = []
    target_q_list = []

    # Iterate over each transition in the sampled batch
    for now_state, action, new_state, reward, done in random_memoy:
        # Assign a high reward if the episode is finished successfully
        if done: 
            target = 10
        else:
            with torch.no_grad():
                target = reward + gamma * policy_dqn(dqn_input(new_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)).max().item()
        
        # Get the current Q-value
        current_q = policy_dqn(dqn_input(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2))
        current_q_list.append(current_q)
        
        # Create a copy of the current Q-values for updating
        target_q = current_q.clone()
        target_q[action] = target # Update the Q-value for the taken action
        target_q_list.append(target_q)

    # Compute the loss between current and target Q-values
    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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

class ReplayMemory():
    def __init__(self, maxlen):
        # Initialize a deque to store the memory, with a fixed maximum length
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        # Add a transition (state, action, reward, next_state) to the memory
        self.memory.append(transition)

    def sample(self, sample_size):
        # Randomly sample a batch of transitions from the memory
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # Return the current length of the memory
        return len(self.memory)

def test_for_save(policy_dqn, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2):
    env2 = gym.make('Acrobot-v1')
    totel_step = 0
    for i in range(10):
        now_state = env2.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0

        # Play one episode
        while not done and step < 300 :
            with torch.no_grad():
                action = policy_dqn(dqn_input(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)).argmax().item()  # Best action
            step += 1
            totel_step += 1
            # Take action and observe result
            new_state, reward, done, truncated, _ = env2.step(action)
            
            now_state = new_state

    return (totel_step / 10)

def train ():

    env = gym.make('Acrobot-v1')
    	
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
    learning_rate = 0.01
    gamma = 0.9

    past_best_save = 300
    save_count = 0
    run_memory = []
    choice_list = ['x'] * 30 + ['y'] * 70 
    totel_steps = 0
    run = 0
    real_run = 0
    memory = ReplayMemory(10000)
    

    while True:
        now_state = env.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0
        run += 1  # Increment the episode counte
        finis = False

        # Play one episode
        while not done and step < 1000:
            # Use the policy network to select the best action
            if random.choice(choice_list) == "x":
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():
                    action = policy_dqn(dqn_input(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)).argmax().item()  # Best action
            step += 1
            totel_steps += 1
            # Take action and observe result
            new_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in memory
            run_memory.append((now_state, action, new_state, reward, done))

            now_state = new_state

        print(step, run, real_run)

        if done: 
            finis = True
        
        # give the 1 point ever steps for successful run
        for now_state, action, new_state, reward, done in run_memory:
            if finis :
                reward = 1
            memory.append((now_state, action, new_state, reward, done))
        run_memory = []

        # update the choice_list
        if step < 1000:
            real_run += 1
            if (real_run % 10) == 0:
                if all(choice == 'y' for choice in choice_list):
                    return 0

                choice_list.remove("x")
                choice_list.append("y")

        # save the best DQN
        if step < int(past_best_save):
            env.close()
            best_save = test_for_save(policy_dqn, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)
            if best_save < past_best_save:
                save_count += 1
                past_best_save = best_save
                torch.save(policy_dqn.state_dict(), "CartPole.pt")

        for _ in range(10):
            optimize(memory, policy_dqn, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2, learning_rate, gamma, finis)

train()