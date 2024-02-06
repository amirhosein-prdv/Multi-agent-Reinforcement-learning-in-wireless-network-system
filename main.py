import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import include.DDPG as DDPG
from Environment import Environment

import copy, json, argparse
from include.dotdic import DotDic
from include.training import *


parser = argparse.ArgumentParser()
parser.add_argument("--method", default="DDPG_MC")  # Policy name
parser.add_argument("--phase", default="train")  # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=1e4, type=int)  # How many time steps purely random policy is run for
parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
parser.add_argument("--expl_noise", default=0.0, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
parser.add_argument('--actor_lr', type=float, default=1e-3)  # learning rate
parser.add_argument('--critic_lr', type=float, default=1e-3)  # learning rate
parser.add_argument('--aux_lr', type=float, default=1e-3)  # learning rate
parser.add_argument('--weight_decay', type=float, default=1e-4)  # weight decay
parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noiseparser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("--gpu_id", default=0, type=int)  # The id of GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('-c1', '--config_path1', default='./Src/configs/config_11.json', type=str, help='path to existing scenarionarios file')
parser.add_argument('-c2', '--config_path2', default='./Src/configs/config_21.json', type=str, help='path to existing options file')
args = parser.parse_args()

file_name_DDPG = "DDPG Training"
if not os.path.exists('%s/' % (file_name_DDPG)):
  os.makedirs('%s/' % ( file_name_DDPG))

sceParam = DotDic(json.loads(open(args.config_path1, 'r').read()))
trial_scenario = copy.deepcopy(sceParam)

optParam = DotDic(json.loads(open(args.config_path2, 'r').read()))
trial_opt = copy.deepcopy(optParam)


K = trial_scenario.K_t + trial_scenario.K_r
L = trial_scenario.L_t + trial_scenario.L_r

# Actions: (Re{w_k}, Im{w_k}), (Re{u_l}, Im{u_l}), (theta_r, theta_t), rho_l
action_size =   (2 * K * trial_scenario.Nt) + \
                (2 * L * trial_scenario.Nt) + \
                (2 * trial_scenario.R * trial_scenario.Mr) + L

# States:  h, g, h_s, g_s, H, f, STR
state_size =    (trial_scenario.Nt * K) + (trial_scenario.Nt * L) + \
                (trial_scenario.Mr * trial_scenario.R * K) + \
                (trial_scenario.Mr * trial_scenario.R * L) + \
                (trial_scenario.Mr * trial_scenario.Nt * trial_scenario.R) + \
                (L * K) + 1

env = Environment(trial_scenario, state_size, action_size)
env_test = Environment(trial_scenario, state_size, action_size)

max_action = 1.0
agent = DDPG.DDPG(state_size, action_size, max_action, args)
training = Training(env, env_test, agent, file_name_DDPG)

trainEpisode = 20
maximum_steps = 200
training_epoch = 10
reward_mem, TR_mem = training.train(trainEpisode, maximum_steps, action_size, training_epoch, args)

print('DDPG_Train=', reward_mem, ';')
print('Mean_DDPG_Train=', np.mean(reward_mem), ';')

testEpisode = 10
maximum_steps = 200
reward_mem_test, TR_mem_test = training.test(testEpisode, maximum_steps, action_size)

print('DDPG_Test=', reward_mem_test, ';')
print('DDPG_Test=', np.mean(reward_mem_test), ';')

error = np.std(TR_mem, axis=0)
plt.plot(range(1, trainEpisode+1), reward_mem)
plt.fill_between(range(1, trainEpisode+1), reward_mem-error, reward_mem+error,
    alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')

error_test = np.std(TR_mem_test, axis=0)
plt.plot(range(1, testEpisode+1), reward_mem_test)
plt.fill_between(range(1, testEpisode+1), reward_mem_test-error, reward_mem_test+error,
    alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.show()