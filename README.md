# Multi-agent-Reinforcement-learning-in-wireless-network-system

This repository provides an implementation of a Full-Duplex Multi STAR-RIS (Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surface) system with Ultra-Reliable Low-Latency Communication (URLLC) services in a wireless network environment. The primary focus of this project is to solve the target optimization problem using Multi-Agent Reinforcement Learning (MARL), specifically the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm.

In addition to MADDPG, this repository includes implementations of various Policy Gradient and Actor-Critic algorithms such as DQN, DDPG, TD3, and SAC, providing a comprehensive toolkit for reinforcement learning research in wireless networks.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Environment Setup](#environment-setup)
- [Citation](#citation)

## Introduction

This project aims to explore the application of Multi-Agent Reinforcement Learning (MARL) for optimizing URLLC service provision in Multi-STAR-RIS assisted full-duplex cellular wireless systems. The environment and optimization problem are based on the work presented in the following paper:

Y. Eghbali, S. K. Taskou, M. R. Mili, M. Rasti and E. Hossain, "Providing URLLC Service in Multi-STAR-RIS Assisted and Full-Duplex Cellular Wireless Systems: A Meta-Learning Approach," in IEEE Communications Letters, doi: 10.1109/LCOMM.2023.3349377.

## Key Features

- **Full-Duplex Multi-STAR-RIS Environment**: Simulate a wireless network environment with Multi-STAR-RIS and URLLC service.
- **Multi-Agent Reinforcement Learning**: Implement the MADDPG algorithm to solve the target optimization problem.
- **Comprehensive Reinforcement Learning Toolkit**: Includes additional algorithms like DQN, DDPG, TD3, and SAC for comparative analysis.
- **Highly Customizable**: Modify the environment and learning algorithms to suit your research needs.

## Environment Setup
### Overview
The environment represents a Multi-STAR-RIS assisted full-duplex (FD) cellular wireless system designed to provide ultra-reliable low-latency communication (URLLC) services. In this system, the key components include a full-duplex base station (BS), multiple STAR-RISs (Simultaneously Transmitting and Reflecting Reconfigurable Intelligent Surfaces), uplink (UL) users, and downlink (DL) users. The goal is to optimize the system's performance by maximizing the total uplink and downlink rates under strict URLLC requirements.


![image](https://github.com/user-attachments/assets/2bb69ddd-d50d-4c91-8f37-63d7ee72765a)

### Environment Dynamics
- **Channel Models:** The environment models various channels including direct BS-user channels, BS-STAR-RIS channels, STAR-RIS-user channels, and self-interference channels at the BS.
- **Signal Processing:**
1. The received signals at the BS and DL users are affected by the beamforming vectors, STAR-RIS coefficients, and self-interference.
2. The SINR (Signal-to-Interference-plus-Noise Ratio) for both UL and DL users is calculated considering multi-user interference and noise.
- **Objective:** The primary objective is to maximize the total system rate (STR) by optimizing the beamforming vectors at the BS, the amplitude attenuations, and phase shifts of the STAR-RISs while meeting the URLLC requirements.

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Gym

## Algorithms
This repository includes the following algorithms:

1. **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)**
MADDPG is an extension of the DDPG algorithm designed for multi-agent environments. Each agent has its own actor and critic networks, but the critic network for each agent considers the actions of all agents in the environment. This allows agents to learn coordinated strategies in complex, multi-agent settings.

![image](https://github.com/user-attachments/assets/63a3e4bf-b9ba-48cf-af3c-bd73ab8cebed)


2. **DQN (Deep Q-Network)**
DQN is a value-based reinforcement learning algorithm that uses a neural network to approximate the Q-value function. It is particularly effective in discrete action spaces and is foundational in the development of many other deep RL algorithms.

3. **DDPG (Deep Deterministic Policy Gradient)**
DDPG is an actor-critic algorithm designed for continuous action spaces. It combines the strengths of DQN and deterministic policy gradients, allowing for efficient learning in high-dimensional, continuous environments.

![image](https://github.com/user-attachments/assets/007ef316-7cc7-4dcc-9ffd-89776e5c3f78)


4. **TD3 (Twin Delayed DDPG)**
TD3 is an improvement over DDPG, addressing some of its limitations such as the overestimation bias in Q-learning. TD3 uses two critic networks to mitigate this bias and delays the update of the target policy, resulting in more stable learning.

![image](https://github.com/user-attachments/assets/e7b4a84a-6d8d-4127-8020-0128dfe540b2)


6. **SAC (Soft Actor-Critic)**
SAC is an off-policy actor-critic algorithm that optimizes a stochastic policy in a maximum entropy framework. This means it aims to maximize both the expected reward and the entropy of the policy, promoting exploration and robustness to different environments.

## Citation
If you find this project useful in your research, please consider citing the original paper:

```bibtex
@article{eghbali2023providing,
  title={Providing URLLC Service in Multi-STAR-RIS Assisted and Full-Duplex Cellular Wireless Systems: A Meta-Learning Approach},
  author={Eghbali, Yashar and Taskou, Sina K and Mili, Mohammadreza and Rasti, Mohammad and Hossain, Ekram},
  journal={IEEE Communications Letters},
  year={2023},
  publisher={IEEE}
}
```
