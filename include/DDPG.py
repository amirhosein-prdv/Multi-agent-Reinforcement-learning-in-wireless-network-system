import torch
import torch.nn as nn
import torch.nn.functional as F
from include.utils import Critic_Network, Hot_Plug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Meta-Critic of Deep Deterministic Policy Gradients (DDPG_MC)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = self.max_action * torch.tanh(self.l3(x))
        return x

class Actor_Feature(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor_Feature, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.actor_feature = Actor_Feature(state_dim, action_dim, max_action).to(device)
        self.actor_feature_target = Actor_Feature(state_dim, action_dim, max_action).to(device)
        self.actor_feature_target.load_state_dict(self.actor_feature.state_dict())
        self.actor_feature_optimizer = torch.optim.Adam(self.actor_feature.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.feature_critic = Critic_Network(128).to(device)
        self.omega_optim = torch.optim.Adam(self.feature_critic.parameters(), lr=args.aux_lr,
                                            weight_decay=args.weight_decay)
        feature_net = nn.Sequential(*list(self.actor_feature.children())[:-1])
        self.lr_actor = args.actor_lr
        self.hotplug = Hot_Plug(feature_net)
        self.max_action = max_action
        self.loss_store = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state_feature = self.actor_feature(state)
        return self.actor(state_feature).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

        for it in range(iterations):
            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(d).to(device)

            # Compute the target Q value
            next_state_feature = self.actor_feature_target(next_state)
            target_Q = self.critic_target(next_state, self.actor_target(next_state_feature))
            target_Q = reward + ((1-done) * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            state_feature = self.actor_feature(state)
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state_feature)).mean()

            # Delayed policy updates
            if it % 1 == 0:
                loss_auxiliary = self.feature_critic(state_feature)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                self.actor_feature_optimizer.zero_grad()

                self.actor_optimizer.step()
                self.actor_feature_optimizer.step()
                self.hotplug.restore()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor_feature.parameters(), self.actor_feature_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Store the loss information
                tmp_loss = []
                tmp_loss.append(critic_loss.item())
                tmp_loss.append(actor_loss.item())
                tmp_loss.append(loss_auxiliary.item())
                # tmp_loss.append(loss_meta.item())
                # self.loss_store.append(tmp_loss)

    def save(self, filename, directory):
        torch.save(self.actor_feature.state_dict(), '%s/%s_actor_feature.pth' % (directory, filename))
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor_feature.load_state_dict(torch.load('%s/%s_actor_feature.pth' % (directory, filename)))
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
