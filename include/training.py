import numpy as np
import include.utils as utils

class Training():
    def __init__(self, env, env_test, agent, path_save):

        self.env = env
        self.env_test = env_test
        self.agent = agent

        self.var_noise = 1
        self.decay_noise = .99993

        self.reward_mem = []
        self.reward_mem_test = []

        self.TR_mem = []
        self.R_DL_mem = []
        self.R_UL_mem = []

        self.TR_mem_test = []
        self.R_DL_mem_test = []
        self.R_UL_mem_test = []

        self.file_name_DDPG = path_save
        self.model_path = '%s' % (path_save)


    def train(self, trainEpisode, maximum_steps, action_size, training_epoch, args):
        
        for e in range(trainEpisode):
            TR = 0
            reward = 0
            done = False
            state = self.env.reset()
            replay_buffer = utils.ReplayBuffer()

            TR_eps = []
            R_UL_eps = []
            R_DL_eps = []

            print(f'Training: \n\t episode {e+1}/{trainEpisode} ...')
            for nstep in range(maximum_steps):

                self.var_noise = self.var_noise * self.decay_noise
                noise = np.random.randn(action_size) * self.var_noise

                state = self.env.state_cal(TR)
                state = abs(state)

                action = self.agent.select_action(state)
                action = action + noise
                action = np.clip(action, -1, 1)

                phi_r, phi_t, rho, w, u = self.env.action_cal(action)
                next_state, reward, done = self.env.step(phi_r, phi_t, rho, w, u)

                replay_buffer.add((state, next_state, action, reward, done))
                state = next_state
                self.agent.train(replay_buffer, training_epoch, args.batch_size, args.discount, args.tau)
                self.agent.save(self.file_name_DDPG, self.model_path)

                Theta_r, Theta_t = self.env.gen_Theta(phi_r, phi_t)
                gamma_DL = self.env.cal_gamma_DL(Theta_r, Theta_t,w,rho)
                gamma_UL = self.env.cal_gamma_UL(Theta_r, Theta_t,rho,w,u)
                R_UL = self.env.cal_R_UL(gamma_UL)
                R_DL = self.env.cal_R_DL(gamma_DL)
                TR = self.env.cal_TR(R_UL, R_DL)

                TR_eps.append(TR)
                R_UL_eps.append(R_UL)
                R_DL_eps.append(R_DL)

                if done:
                    self.TR_mem.append(TR_eps)
                    self.R_UL_mem.append(R_UL_eps)
                    self.R_DL_mem.append(R_DL_eps)
                    print(f'\t done in step {nstep}')
                    break

            if len(self.TR_mem) > 0:
                self.reward_mem.append(np.mean(TR_eps))
            
            print('\t Objective function is ', np.mean(self.reward_mem))

        return self.reward_mem, self.TR_mem

    
    def test(self, testEpisode, maximum_steps, action_size):
        for e in range(testEpisode):
            TR = 0
            reward = 0
            done = False
            state = self.env_test.reset()
            self.agent.load(self.file_name_DDPG, self.model_path)

            TR_eps = []
            R_UL_eps = []
            R_DL_eps = []
            
            print(f'Testing: \n\t episode {e+1}/{testEpisode} ...')
            for nstep in range(maximum_steps):
                self.var_noise = self.var_noise * self.decay_noise
                noise = (np.random.randn(action_size) * self.var_noise)

                state = self.env_test.state_cal(TR)
                state = abs(state)
                action = self.agent.select_action(state)
                action = action + noise
                action = np.clip(action, -1, 1)

                phi_r, phi_t, rho, w, u = self.env_test.action_cal(action)
                new_state, reward, done = self.env_test.step(phi_r, phi_t, rho, w, u)
                state = new_state

                Theta_r, Theta_t = self.env_test.gen_Theta(phi_r, phi_t)
                gamma_DL = self.env_test.cal_gamma_DL(Theta_r, Theta_t, w, rho)
                gamma_UL = self.env_test.cal_gamma_UL(Theta_r, Theta_t, rho, w, u)
                R_UL = self.env_test.cal_R_UL(gamma_UL)
                R_DL = self.env_test.cal_R_DL(gamma_DL)
                TR = self.env_test.cal_TR(R_UL, R_DL)

                TR_eps.append(TR)
                R_UL_eps.append(R_UL)
                R_DL_eps.append(R_DL)
                
                if done:
                    self.TR_mem_test.append(TR_eps)
                    self.R_UL_mem_test.append(R_UL_eps)
                    self.R_DL_mem_test.append(R_DL_eps)
                    print(f'\t done in step {nstep}')
                    break
            
            self.reward_mem_test.append(np.mean(TR_eps))

            print('\t Objective function is ', np.mean(self.reward_mem_test))

        return self.reward_mem_test, self.TR_mem_test