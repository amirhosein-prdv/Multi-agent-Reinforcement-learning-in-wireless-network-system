import copy
import random
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Replay_buffer():
    
    # Expects tuples of 
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0
    
    def reset(self):
        self.buffer = []
        self.ptr = 0

    def push(self, data):
        if len(self.buffer) == self.max_size:
            self.buffer[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.buffer.append(data) # (state, next_state, action, reward, done)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []

        for i in ind:
            st, n_st, act, rew, dn = self.buffer[i]
            state.append(np.array(st, copy=False))
            next_state.append(np.array(n_st, copy=False))
            action.append(np.array(act, copy=False))
            reward.append(np.array(rew, copy=False))
            done.append(np.array(dn, copy=False))

        return np.array(state), np.array(next_state), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1[0]), 
            nn.ReLU(), 
            nn.Linear(hidden1[0], hidden1[1]), 
            nn.ReLU(), 
            nn.Linear(hidden1[1], hidden1[2]), 
            nn.ReLU(), 
            nn.Linear(hidden1[2], action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden2[0]), 
            nn.ReLU(), 
            nn.Linear(hidden2[0], hidden2[1]), 
            nn.ReLU(), 
            nn.Linear(hidden2[1], hidden2[2]), 
            nn.ReLU(), 
            nn.Linear(hidden2[2], 1),
        )
        
    def forward(self, state, action):
        return self.net(torch.cat((state, action), 1))
    

class DDPG(object):
    def __init__(self, state_dim, action_dim, args):

        self.args = args
        
        self.replay_buffer = Replay_buffer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, self.args.actor_hidden).to(self.device)
        self.actor_target = Actor(state_dim, action_dim,  self.args.actor_hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)

        self.critic = Critic(state_dim, action_dim, self.args.critic_hidden).to(self.device)
        self.critic_target = Critic(state_dim, action_dim,  self.args.critic_hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)       

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, update_iteration=50):

        for it in range(update_iteration):
            # For each Sample in replay buffer batch
            state, next_state, action, reward, done = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1-done) * self.args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()            

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            
           
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), self.args.model_path + '/actor.pth')
        torch.save(self.critic.state_dict(), self.args.model_path + '/critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(self.args.model_path + '/actor.pth'))
        self.critic.load_state_dict(torch.load(self.args.model_path + '/critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
