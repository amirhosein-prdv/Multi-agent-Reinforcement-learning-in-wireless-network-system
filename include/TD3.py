import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def reset(self):
        self.buffer = []
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        self.buffer.append(transition) # (state, next_state, action, reward, done)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, next_state, action, reward, done = [], [], [], [], []
        
        for i in indexes:
            s, s_, a, r, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(next_state), np.array(action), np.array(reward), np.array(done)
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, actor_hidden):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, actor_hidden[0])
        self.l2 = nn.Linear(actor_hidden[0], actor_hidden[1])
        self.l3 = nn.Linear(actor_hidden[1], action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.max_action = max_action
        
    def forward(self, state):
        x = self.relu(self.l1(state))
        x = self.relu(self.l2(x))
        x = self.tanh(self.l3(x)) 
        x = x * self.max_action
        return x
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, critic_hidden):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, critic_hidden[0])
        self.l2 = nn.Linear(critic_hidden[0], critic_hidden[1])
        self.l3 = nn.Linear(critic_hidden[1], 1)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
class TD3:
    def __init__(self, state_dim, action_dim, max_action, args):
        
        self.args = args

        self.actor = Actor(state_dim, action_dim, max_action, self.args.actor_hidden).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, self.args.actor_hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.lr)
        
        self.critic_1 = Critic(state_dim, action_dim, self.args.critic_hidden).to(device)
        self.critic_1_target = Critic(state_dim, action_dim, self.args.critic_hidden).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.args.lr)
        
        self.critic_2 = Critic(state_dim, action_dim, self.args.critic_hidden).to(device)
        self.critic_2_target = Critic(state_dim, action_dim, self.args.critic_hidden).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.args.lr)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, n_iter):
        
        for i in range(n_iter):
            state, next_state, action_, reward, done = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((self.args.batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((self.args.batch_size,1)).to(device)
            
            # Select next action
            noise = torch.ones_like(torch.FloatTensor(action_)).data.normal_(0, self.args.policy_noise).to(device)
            noise = noise.clamp(-self.args.noise_clip, self.args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * self.args.gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            if i % self.args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (self.args.polyak * target_param.data) + ((1-self.args.polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (self.args.polyak * target_param.data) + ((1-self.args.polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (self.args.polyak * target_param.data) + ((1-self.args.polyak) * param.data))
                    
                
    def save(self, directory=None, name='TD3'):
        if directory == None:
            directory = self.args.model_path
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))
        
    def load(self, directory=None, name='TD3'):
        if directory == None:
            directory = self.args.model_path
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
    def load_actor(self, directory=None, name='TD3'):
        if directory == None:
            directory = self.args.model_path
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
