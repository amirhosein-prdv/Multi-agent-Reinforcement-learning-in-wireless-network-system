import numpy as np
import include.utils as utils

class Training():
    def __init__(self, T_train, T_test, env, env_test, agent, path_save):

        self.T_train = T_train
        self.T_test = T_test
        self.TR = 0
        self.reward_mem = []


        self.TR_test = 0
        self.TR_mem_test = []
        self.reward_mem_test = []

        self.env = env
        self.env_test = env_test
        self.agent = agent


        self.var_noise = 1
        self.decay_noise = .99993

        self.TR_mem_UL = []
        self.TR_mem_DL = []
        self.TR_mem_UL_test = []
        self.TR_mem_DL_test = []

        self.replay_buffer = utils.ReplayBuffer()

        self.file_name_DDPG = path_save
        self.model_path = '%s' % (path_save)


    def train(self, T_train, maximum_steps, action_size, training_steps, args):
        
        for e in range(T_train):
            self.TR_mem = []
            state = self.env.reset()
            reward = 0
            reward_t = 0
            nstep = 0
            flag = False

            print(f'Training: epoch {e}/T_train ...')

            while not flag and nstep <= maximum_steps:
                nstep += 1
                self.var_noise = self.var_noise * self.decay_noise
                # ***** DDPG *****
                state = self.env.state_cal(self.TR)
                state = abs(state)
                action = self.agent.select_action(state)
                action = action + (np.random.randn(action_size) * self.var_noise)
                action = np.clip(action, -1, 1)

                w, u, rho, phi_r, phi_t = self.env.action_cal(action)

                Theta_r, Theta_t = self.env.gen_Theta( phi_r, phi_t)
                gamma_DL = self.env.cal_gamma_DL(Theta_r, Theta_t,w,rho)
                gamma_UL = self.env.cal_gamma_UL(Theta_r, Theta_t,rho,w,u)
                R_UL = self.env.cal_R_UL(gamma_UL)
                R_DL = self.env.cal_R_DL(gamma_DL)
                self.TR = self.env.cal_TR(R_UL, R_DL)

                new_state, reward, done = self.env.step(phi_r, phi_t, rho, w, u)


                if done:
                    self.TR_mem.append(self.TR)
                    self.TR_mem_UL.append(R_UL)
                    self.TR_mem_DL.append(R_DL)
                    reward += self.TR
                    reward_t += self.TR
                else:
                    reward_t += 0

                self.replay_buffer.add((state, new_state, action, reward, done))
                state = new_state
                self.agent.train(self.replay_buffer, training_steps, args.batch_size, args.discount, args.tau)
                self.agent.save(self.file_name_DDPG, self.model_path)

                if done:
                  flag = True
            
            if len(self.TR_mem) > 0:
                self.reward_mem.append(np.mean(self.TR_mem))
            
            # print('Objective function is ', self.reward_mem,)
            print('Objective function is ', np.mean(self.reward_mem))

        return self.reward_mem, self.TR_mem

    
    def test(self, T_test, action_size):
        for e_test in range(T_test):
            maximum_steps=500
            state_test = self.env_test.reset()
            reward_test = 0
            reward_t_test = 0
            nstep_test = 0
            flag_test = False

            print(f'Testing: epoch {e_test}/T_train ...')

            while  nstep_test <= maximum_steps:
                nstep_test += 1
                self.var_noise = self.var_noise * self.decay_noise
                # *****  DDPG *****
                self.agent.load(self.file_name_DDPG, self.model_path)
                state_test = self.env_test.state_cal(self.TR_test)
                state_test = abs(state_test)
                action_test = self.agent.select_action(state_test)
                action_test = action_test + (np.random.randn(action_size) * self.var_noise)
                action_test = np.clip(action_test, -1, 1)

                w_test, u_test, rho_test, phi_r_test, phi_t_test = self.env_test.action_cal(action_test)
                Theta_r_test, Theta_t_test = self.env_test.gen_Theta(phi_r_test, phi_t_test)
                gamma_DL_test = self.env_test.cal_gamma_DL(Theta_r_test, Theta_t_test, w_test, rho_test)
                gamma_UL_test = self.env_test.cal_gamma_UL(Theta_r_test, Theta_t_test, rho_test, w_test, u_test)
                R_UL_test = self.env_test.cal_R_UL(gamma_UL_test)
                R_DL_test = self.env_test.cal_R_DL(gamma_DL_test)

                self.TR_test = self.env_test.cal_TR(R_UL_test, R_DL_test)

                new_state_test, reward_test, done_test = self.env_test.step(phi_r_test, phi_t_test, rho_test, w_test, u_test)
                
                if done_test:
                    self.TR_mem_test.append(self.TR_test)
                    self.TR_mem_UL_test.append(R_UL_test)
                    self.TR_mem_DL_test.append(R_DL_test)
                    reward_test += self.TR_test
                    reward_t_test += self.TR_test
                else:
                    reward_t_test += 0

                state_test = new_state_test
                if done_test:
                    flag_test = True
            
            self.reward_mem_test.append(np.mean(self.TR_mem_test))

            # print('Objective_Function_Test= ', self.reward_mem_test, ';')
            print('Objective function is ', np.mean(self.reward_mem_test))

        return self.reward_mem_test, self.TR_mem_test