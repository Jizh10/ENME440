import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


env = gym.make('Acrobot-v1')
env.reset()

# hyper parameters
gamma = 0.95
epsilon_init = 0.9
epsilon_end = 0.001
hidden_size = 12
hidden_layers = 1
learning_rate = 0.0005

# training sets for comparison
training_sets = [1000]
ep_rewards = np.array([])

# extract input and output based on the model
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.n
#device = torch.device("cpu")

# define nn class Q-learning Net
class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, states):
        x = self.relu(self.fc1(states))
        return self.out(x)
    
class ExpReplayBuffer():
    
    def __init__(self):
        self.buffer = []
        
    def add(self, state, action, reward, next_states, done):
        self.buffer.append((state, action, reward, next_states, done))
    
    def __len__(self):
        return len(self.buffer)
    
    def sampleBatch(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        indices = np.random.choice(len(self.buffer), batch_size)
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(np.array(state))
            actions.append(np.array(action))
            rewards.append(reward)
            next_states.append(np.array(next_state))
            dones.append(done)
        
        states = torch.as_tensor(np.array(states))
        actions = torch.as_tensor(np.array(actions))
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32))
        next_states = torch.as_tensor(np.array(next_states))
        dones = torch.as_tensor(np.array(dones, dtype=np.float32))
        
        return states, actions, rewards, next_states, dones        

# helper function for epsilon exploration
def epsilon_greedy_action(state, epsilon):
    
    if (np.random.random(1) < epsilon):
        # randomly return -1, 0, or 1
        return np.random.randint(3)
    else:
        # return -1, 0, or 1 based on the Q value obtained from the nn
        qs = main_net(state).cpu().data.numpy()
        return np.argmax(qs)


frame = 0
exp_buffer = ExpReplayBuffer()
batch_size = 40
# training code
for trainings in training_sets:
    # set up the nn, loss function, and optimizer
    main_net = DQN(input_size=num_inputs, output_size=num_outputs, hidden_size=hidden_size, hidden_layers=hidden_layers)
    target_net = DQN(input_size=num_inputs, output_size=num_outputs, hidden_size=hidden_size, hidden_layers=hidden_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(main_net.parameters(), lr=learning_rate, weight_decay=0)
    for epoch in range(trainings):
        # reset the model and init states, epoch reward, if the process is done or not, and epsilon
        curr_state = env.reset().astype(np.float32)
        ep_reward, done = 0, False
        epsilon = (epsilon_init - epsilon_end)*(1 - epoch/trainings) + epsilon_end
            
        while not done:
            
            # convert the prev state from numpyto tensor
            state_in = torch.from_numpy(np.expand_dims(curr_state, axis=0))
            # get the next action to take
            action = epsilon_greedy_action(state_in, epsilon)
            # take the action and get the resulting state and reward
            next_state, reward, done, _ = env.step(action)
            # convert the state to desired numpy type
            next_state = next_state.astype(np.float32)
            ep_reward += reward
            
            # save to the experience replay buffer
            exp_buffer.add(curr_state, action, reward, next_state, done)
            curr_state = next_state
            
            frame += 1
            
            # train the nn net
            if len(exp_buffer) > batch_size:
                states, actions, rewards, next_states, dones = exp_buffer.sampleBatch(batch_size)
                max_next_qs = target_net(next_states).max(-1).values
                target = rewards + (1.0 - dones) * gamma * max_next_qs
                qs = main_net(states)
                
                actions_mask = []
                for i, action in enumerate(actions):
                    mask = np.array([0, 0, 0])
                    mask[action.numpy()] = 1
                    actions_mask.append(mask)
    
                actions_mask = torch.as_tensor(np.array(actions_mask))
                
                #action_masks = F.one_hot(actions.long(), num_outputs)
                qs_action = (actions_mask * qs).sum(dim=-1)
                loss = criterion(qs_action, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if frame % 1000 == 0:
                target_net.load_state_dict(main_net.state_dict())
            
            if done:
                if (epoch+1)%10 == 0:
                    print('epoch %d, epsilon: %f, nn loss: %f, total reward: %d' % (epoch+1, epsilon, loss.item(), ep_reward))
                ep_rewards = np.append(ep_rewards, ep_reward)
            
            
            if epoch+1 == trainings:
                env.render()
                
    env.close() 
    
    plt.figure(1)
    plt.plot(np.arange(trainings), ep_rewards)
            
         