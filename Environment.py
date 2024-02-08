import numpy as np
from scipy import special
import random

class Environment:
    def __init__(self, scenario, state_dim, action_dim):
        self.e = np.e
        self.pi = np.pi
        self.scenario = scenario

        # UL-DL users
        self.K = self.scenario.K_t + self.scenario.K_r
        self.L = self.scenario.L_t + self.scenario.L_r

        # Rician param.
        self.Rician_factor = 4
        self.variance = np.sqrt(1 / (self.Rician_factor + 1))
        self.mean = np.sqrt(self.Rician_factor / (self.Rician_factor + 1))

        # user selection param.
        self.beta_UL = np.ones(self.L)
        self.beta_DL = np.ones(self.K)

        # state-action
        self.state_size = state_dim
        self.action_size = action_dim
        self.action_space = []
        for i in range(self.action_size):
            self.action_space.append(random.uniform(0, 1))
        
        # generate DL symbols and channels
        self.s = self.gen_s()

        self.h = self.gen_h()
        self.g = self.gen_g()
        self.f = self.gen_f()
        self.H = self.gen_H()
        self.g_s = self.gen_g_s()
        self.h_s = self.gen_h_s()
    
    def set_seed(self, seed):
        np.random.seed(seed)
        
    # i.i.d. information symbol for the k'th DL user
    def gen_s(self):
        s = np.random.randn(self.K) + 1j * np.random.randn(self.K)
        return s

    # channel vector between BS and k_b'th DL user (Rayleigh)
    def gen_h(self):
        h = np.random.randn(self.scenario.Nt, self.K) + 1j * np.random.randn(self.scenario.Nt, self.K)
        return h
    
    # channel vector between l_b'th UL and BS (Rayleigh)
    def gen_g(self):
        g = np.random.randn(self.scenario.Nt, self.L) + 1j * np.random.randn(self.scenario.Nt, self.L)
        return g
    
    # channel matrix between BS and q'th RIS (Rician)
    def gen_H(self):
        H = ((self.mean * np.random.randn(self.scenario.Mr, self.scenario.Nt, self.scenario.R))) + (
                self.variance * np.random.randn(self.scenario.Mr, self.scenario.Nt, self.scenario.R))
        return H

    # channel between the l_b'th UL user and k_b'th DL user (Rician)
    def gen_f(self):
        f = ((self.mean * np.random.randn(self.L, self.K))) + (
                self.variance * np.random.randn(self.L, self.K))
        return f

    # channel vector between k_b'th DL user and q'th RIS (Rician)
    def gen_h_s(self):
        h_s = ((self.mean * np.random.randn(self.scenario.Mr, self.scenario.R, self.K))) + (
                self.variance * np.random.randn(self.scenario.Mr, self.scenario.R, self.K))
        return h_s
   
    # channel vector between the l_b'th UL user and q'th RIS (Rician)
    def gen_g_s(self):
        g_s = ((self.mean * np.random.randn(self.scenario.Mr, self.L, self.scenario.R))) + (
                self.variance * np.random.randn(self.scenario.Mr, self.L, self.scenario.R))
        return g_s

    # coefficient matrix of the qth RIS
    def gen_Theta(self, phi_r, phi_t):
        Theta_r = np.zeros([self.scenario.R, self.scenario.Mr, self.scenario.Mr], dtype=complex)
        Theta_t = np.zeros([self.scenario.R, self.scenario.Mr, self.scenario.Mr], dtype=complex)
        temp_Diag_r = np.zeros([self.scenario.Mr, self.scenario.Mr], dtype=complex)
        temp_Diag_t = np.zeros([self.scenario.Mr, self.scenario.Mr], dtype=complex)

        for r in range(self.scenario.R):
            phi_rr = phi_r[r, :]
            phi_tt = phi_t[r, :]
            for mr in range(self.scenario.Mr):
                phi_r_mr = phi_rr[mr]
                phi_t_mr = phi_tt[mr]
                temp_Diag_r[mr, mr] = self.e ** (phi_r_mr)
                temp_Diag_t[mr, mr] = self.e ** (phi_t_mr)
                Theta_r[r, :, :] = temp_Diag_r
                Theta_t[r, :, :] = temp_Diag_t
        return Theta_r, Theta_t

    def cal_gamma_UL(self, Theta_r, Theta_t, rho, w, u):
        g = self.g
        g_s = self.g_s
        H = self.H
        gamma_UL = np.zeros([self.L])

        for l in range(self.scenario.L_r):
            num_r = 0
            den1_r = 0
            den2_r = 0
            den3_r = 0
            for r in range(self.scenario.R):
                Tg_s_r = np.matmul(Theta_r[r, :, :], g_s[:, l, r])
                H_H_r = np.transpose(np.conjugate(H[:, :, r]))
                num_r += np.matmul(H_H_r, Tg_s_r)
                # print('num=', num.shape)
                xx_r = np.transpose(np.conjugate(u[:, l]))
                # print('xx=',xx.shape)
            num_r += g[:, l]
            num_r = (np.linalg.norm(num_r * xx_r)) ** 2 * rho[l]
            for ll in range(self.scenario.L_r):
                if ll != l:
                    for r in range(self.scenario.R):
                        Tg_s_r = np.matmul(Theta_r[r, :, :], g_s[:, ll, r])
                        H_H_r = np.transpose(np.conjugate(H[:, :, r]))
                        den1_r += np.matmul(H_H_r, Tg_s_r)

                den1_r = ((np.linalg.norm(g[:, ll] * np.transpose(np.conjugate(u[:, l])))) ** 2) * rho[l]
                den2_r += den1_r
            for ll in range(self.scenario.L_r, self.scenario.L_r + self.scenario.L_t):
                if ll != l:
                    for r in range(self.scenario.R):
                        Tg_s_t = np.matmul(Theta_t[r, :, :], g_s[:, ll, r])
                        H_H_t = np.transpose(np.conjugate(H[:, :, r]))
                        den1_r += np.matmul(H_H_t, Tg_s_t)

                den1_r = ((np.linalg.norm(g[:, ll] * np.transpose(np.conjugate(u[:, l])))) ** 2) * rho[l]
                den2_r += den1_r
            for k in range(self.K):
                den3_r += (np.linalg.norm(w[:, k])) ** 2

            den3_r = den3_r * ((np.linalg.norm(u[:, l])) ** 2) * (self.scenario.sigma_H ** 2)
            den3_r = den3_r + (self.scenario.sigma_UL ** 2) * (np.linalg.norm(u[:, l])) ** 2
            gamma_UL[l] = num_r / (den2_r + den3_r)

        for l in range(self.scenario.L_r, self.scenario.L_t + self.scenario.L_r):
            num_t = 0
            den1_t = 0
            den2_t = 0
            den3_t = 0
            for r in range(self.scenario.R):
                Tg_s_t = np.matmul(Theta_t[r, :, :], g_s[:, l, r])
                H_H_t = np.transpose(np.conjugate(H[:, :, r]))
                num_t += np.matmul(H_H_t, Tg_s_t)
                # print('num=', num.shape)
                xx_t = np.transpose(np.conjugate(u[:, l]))
                # print('xx=',xx.shape)
            num_t += g[:, l]
            num_t = (np.linalg.norm(num_t * xx_t)) ** 2 * rho[l]
            for ll in range(self.scenario.L_r):
                if ll != l:
                    for r in range(self.scenario.R):
                        Tg_s_r = np.matmul(Theta_r[r, :, :], g_s[:, ll, r])
                        H_H_r = np.transpose(np.conjugate(H[:, :, r]))
                        den1_t += np.matmul(H_H_r, Tg_s_r)

                den1_t = ((np.linalg.norm(g[:, ll] * np.transpose(np.conjugate(u[:, l])))) ** 2) * rho[l]
                den2_t += den1_t
            for ll in range(self.scenario.L_r, self.scenario.L_r + self.scenario.L_t):
                if ll != l:
                    for r in range(self.scenario.R):
                        Tg_s_t = np.matmul(Theta_t[r, :, :], g_s[:, ll, r])
                        H_H_t = np.transpose(np.conjugate(H[:, :, r]))
                        den1_t += np.matmul(H_H_t, Tg_s_t)

                den1_t = ((np.linalg.norm(g[:, ll] * np.transpose(np.conjugate(u[:, l])))) ** 2) * rho[l]
                den2_t += den1_t
            for k in range(self.K):
                den3_t += (np.linalg.norm(w[:, k])) ** 2

            den3_t = den3_t * ((np.linalg.norm(u[:, l])) ** 2) * (self.scenario.sigma_H ** 2)
            den3_t = den3_t + (self.scenario.sigma_UL ** 2) * (np.linalg.norm(u[:, l])) ** 2
            gamma_UL[l] = num_t / (den2_t + den3_t)
        return abs(gamma_UL)

    def cal_gamma_DL(self, Theta_r, Theta_t, w, rho):
        h = self.h
        f = self.f
        g_s = self.g_s
        h_s = self.h_s
        H = self.H

        gamma_DL = np.zeros([self.K], dtype=complex)
        # Theta=Theta.reshape([self.scenario.Mr, self.scenario.R, self.scenario.Mr])

        for k in range(self.scenario.K_r):
            num_r = 0
            den1_r = 0
            den2_r = 0
            den3_r = 0
            for r in range(self.scenario.R):
                TH_r = np.matmul(Theta_r[r, :, :], H[:, :, r])
                h_s_H_r = np.transpose(np.conjugate(h_s[:, r, k]))
                num_r += np.matmul(h_s_H_r, TH_r)

            num_r += np.transpose(np.conjugate(h[:, k]))
            num_r = np.transpose(np.conjugate(num_r))
            num_r = np.matmul(num_r, w[:, k])
            num_r = (np.linalg.norm(num_r)) ** 2

            for kk in range(self.scenario.K_r):
                if kk != k:
                    h_k_H_r = np.transpose(np.conjugate(h[:, k]))
                    h_k_HW_r = np.matmul(h_k_H_r, w[:, kk])
                    den1_r += (np.linalg.norm(h_k_HW_r)) ** 2
            for kk in range(self.scenario.K_r, self.scenario.K_r + self.scenario.K_t):
                if kk != k:
                    h_k_H_t = np.transpose(np.conjugate(h[:, k]))
                    h_k_HW_t = np.matmul(h_k_H_t, w[:, kk])
                    den1_r += (np.linalg.norm(h_k_HW_t)) ** 2

            for l in range(self.scenario.L_r):
                den2_r += ((np.linalg.norm(f[l, k])) ** 2) * rho[l]
                for r in range(self.scenario.R):
                    Tg_r = np.matmul(Theta_r[r, :, :], g_s[:, l, r])
                    hsk_r = h_s[:, r, k]
                den3_r += np.matmul(hsk_r, Tg_r)
                den3_r += ((np.linalg.norm(den3_r)) ** 2) * rho[l]

            for l in range(self.scenario.L_r, self.scenario.L_r + self.scenario.L_t):
                den2_r += ((np.linalg.norm(f[l, k])) ** 2) * rho[l]
                for r in range(self.scenario.R):
                    Tg_r = np.matmul(Theta_t[r, :, :], g_s[:, l, r])
                    hsk_r = h_s[:, r, k]
                den3_r += np.matmul(hsk_r, Tg_r)
                den3_r += ((np.linalg.norm(den3_r)) ** 2) * rho[l]

            gamma_DL[k] = num_r / (den1_r + den2_r + den3_r + self.scenario.sigma_DL)

        for k in range(self.scenario.K_r, self.scenario.K_r + self.scenario.K_t):
            num_t = 0
            den1_t = 0
            den2_t = 0
            den3_t = 0
            for r in range(self.scenario.R):
                TH_t = np.matmul(Theta_t[r, :, :], H[:, :, r])
                h_s_H_t = np.transpose(np.conjugate(h_s[:, r, k]))
                num_t += np.matmul(h_s_H_t, TH_t)

            num_t += np.transpose(np.conjugate(h[:, k]))
            num_t = np.transpose(np.conjugate(num_t))
            num_t = np.matmul(num_t, w[:, k])
            num_t = (np.linalg.norm(num_t)) ** 2

            for kk in range(self.scenario.K_r):
                if kk != k:
                    h_k_H_t = np.transpose(np.conjugate(h[:, k]))
                    h_k_HW_t = np.matmul(h_k_H_t, w[:, kk])
                    den1_t += (np.linalg.norm(h_k_HW_t)) ** 2
            for kk in range(self.scenario.K_r, self.scenario.K_r + self.scenario.K_t):
                if kk != k:
                    h_k_H_t = np.transpose(np.conjugate(h[:, k]))
                    h_k_HW_t = np.matmul(h_k_H_t, w[:, kk])
                    den1_t += (np.linalg.norm(h_k_HW_t)) ** 2

            for l in range(self.scenario.L_r):
                den2_t += ((np.linalg.norm(f[l, k])) ** 2) * rho[l]
                for r in range(self.scenario.R):
                    Tg_t = np.matmul(Theta_r[r, :, :], g_s[:, l, r])
                    hsk_t = h_s[:, r, k]
                den3_t += np.matmul(hsk_t, Tg_t)
                den3_t += ((np.linalg.norm(den3_t)) ** 2) * rho[l]

            for l in range(self.scenario.L_r, self.scenario.L_r + self.scenario.L_t):
                den2_t += ((np.linalg.norm(f[l, k])) ** 2) * rho[l]
                for r in range(self.scenario.R):
                    Tg_t = np.matmul(Theta_t[r, :, :], g_s[:, l, r])
                    hsk_t = h_s[:, r, k]
                den3_t += np.matmul(hsk_t, Tg_t)
                den3_t += ((np.linalg.norm(den3_t)) ** 2) * rho[l]

            gamma_DL[k] = num_t / (den1_t + den2_t + den3_t + self.scenario.sigma_DL)

        return abs(gamma_DL)

    def cal_R_DL(self, gamma_DL):
        R_DL = np.zeros(self.K)
        V_DL = np.zeros(self.K)

        blocklength = ((self.scenario.W)) * (self.scenario.transmission_duration) / 2
        Qx = 0.5 - 0.5 * special.erf(self.scenario.decoding_error / np.sqrt(2))
        Qx_inv = special.erfcinv(Qx)

        for k in range(self.K):
            V_DL[k] = ((np.log2(np.exp(1))) ** 2) * (1 - ((1 + gamma_DL[k]) ** -2))
            R_DL[k] = self.scenario.W * (np.log2(1 + gamma_DL[k]) - (np.sqrt(V_DL[k] / blocklength) * Qx_inv))
        return R_DL

    def cal_R_UL(self, gamma_UL):
        R_UL = np.zeros(self.L)
        V_UL = np.zeros(self.L)

        blocklength = ((self.scenario.W)) * (self.scenario.transmission_duration) / 2
        Qx = 0.5 - 0.5 * special.erf(self.scenario.decoding_error / np.sqrt(2))
        Qx_inv = special.erfcinv(Qx)

        for l in range(self.L):
            V_UL[l] = ((np.log2(np.exp(1))) ** 2) * (1 - ((1 + gamma_UL[l]) ** -2))
            R_UL[l] = self.scenario.W * (np.log2(1 + gamma_UL[l]) - (np.sqrt(V_UL[l] / blocklength) * Qx_inv))
        return R_UL

    def cal_TR(self, R_UL, R_DL):
        TR1 = 0
        TR2 = 0
        for l in range(self.L):
            TR1 += self.beta_UL[l] * R_UL[l]
        TR1 = TR1 * (1 - self.scenario.alpha)
        for k in range(self.K):
            TR2 += self.beta_DL[k] * R_DL[k]
        TR2 = TR2 * self.scenario.alpha
        TR = TR1 + TR2
        return TR

    def state_cal(self, TR):
        h = self.h
        g = self.g
        f = self.f
        h_s = self.h_s
        g_s = self.g_s
        H = self.H

        state = np.zeros(self.state_size, dtype=complex)

        start = 0  # h
        end = (self.scenario.Nt * self.K)
        state[start:end] = np.reshape(h, self.scenario.Nt * self.K)

        start = end  # g
        end = end + (self.scenario.Nt * self.L)
        state[start:end] = np.reshape(g, self.scenario.Nt * self.L)

        start = end  # h_s
        end = end + (self.scenario.Mr * self.scenario.R * self.K)
        state[start:end] = np.reshape(h_s, self.scenario.Mr * self.K * self.scenario.R)

        start = end  # g_s
        end = end + (self.scenario.Mr * self.L * self.scenario.R)
        state[start:end] = np.reshape(g_s, self.scenario.Mr * self.L * self.scenario.R)

        start = end  # H
        end = end + (self.scenario.Mr * self.scenario.Nt * self.scenario.R)
        state[start:end] = np.reshape(H, self.scenario.Mr * self.scenario.Nt * self.scenario.R)

        start = end  # f
        end = end + (self.L * self.K)
        state[start:end] = np.reshape(f, self.L * self.K)

        start = end
        end = end + 1
        state[start:end] = TR
        return state

    def action_cal(self, action):
        start = 0
        end = self.K * self.scenario.Nt
        w = np.zeros([self.scenario.Nt, self.K], dtype=complex)
        w1 = (action[start:end])

        start = end
        end = end + self.K * self.scenario.Nt
        w2 = (action[start:end])

        for n in range(self.scenario.Nt):
            for k in range(self.K):
                w[:, k] = w1[k] + 1j * w2[k]

        w = np.reshape(w, [self.scenario.Nt, self.K]) * 0.7

        start = end
        end = end + self.L * self.scenario.Nt
        u = np.zeros([self.scenario.Nt, self.L], dtype=complex)
        u1 = (action[start:end])

        start = end
        end = end + self.L * self.scenario.Nt
        u2 = (action[start:end])

        for n in range(self.scenario.Nt):
            for l in range(self.L):
                u[:, l] = u1[l] + 1j * u2[l]
    
        u = np.reshape(u, [self.scenario.Nt, self.L]) * 0.7

        start = end
        end = end + self.L
        rho = action[start:end]
        rho = ((action[start:end] + 1) / 2) * 0.009

        start = end
        end = end + self.scenario.R * self.scenario.Mr
        phi_r = ((action[start:end] + 1) / 2) 
        phi_r = np.reshape(phi_r, [self.scenario.R, self.scenario.Mr]) * 2 * self.pi

        start = end
        end = end + self.scenario.R * self.scenario.Mr
        phi_t = ((action[start:end] + 1) / 2)
        phi_t = np.reshape(phi_t, [self.scenario.R, self.scenario.Mr]) * 2 * self.pi

        return phi_r, phi_t, rho, w, u

    def reset(self):  # Reset the states
        s = np.zeros((self.state_size))
        return s.reshape(-1)

    def step(self, phi_r, phi_t, rho, w, u):
    
        done = False

        Theta_r, Theta_t = self.gen_Theta(phi_r, phi_t)

        gamma_DL = self.cal_gamma_DL(Theta_r, Theta_t, w, rho)
        gamma_UL = self.cal_gamma_UL(Theta_r, Theta_t, rho, w, u)
        R_DL = self.cal_R_DL(gamma_DL)
        R_UL = self.cal_R_UL(gamma_UL)

        com_w = 0
        check_w = 0
        for k in range(self.K):
            com_w += (np.linalg.norm(w[:, k])) ** 2
            if com_w <= self.scenario.P_T_BS:
                check_w = 1

        check_rho = 0
        for l in range(self.L):
            if rho[l] <= self.scenario.P_l_max:
                check_rho += 1

        check_R_UL = 0
        for l in range(self.L):
            if R_UL[l] >= self.scenario.Rmin_UL:
                check_R_UL += 1
        check_R_DL = 0
        for k in range(self.K):
            if R_DL[k] >= self.scenario.Rmin_DL:
                check_R_DL += 1

        TR = self.cal_TR(R_UL, R_DL)
        next_s = self.state_cal(TR)

        if check_w == 1:
            if check_rho == self.L and check_R_UL == self.L and check_R_DL == self.K:
                reward = TR
                done = True
            else:
                reward = 0
        else:
            reward = 0
        
        return next_s, reward, done
