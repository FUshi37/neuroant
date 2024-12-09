import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import time
from frequency_estimator import *
import json


class CPGs_1:
    def __init__(self,theta, dt,b):
        self.a = 100.0
        self.b = b  # 占空比
        self.w_a = 2* np.pi
        self.w_stance = self.b * self.w_a
        self.w_swing = (1 - self.b) * self.w_a
        self.k1 = 2  # 不同腿之间的扩散系数
        self.k2 = 2
        self.p = 100  # 收敛速度
        self.u = 1
        self.dt = dt
        self.theta=theta

        self.dzdt = np.zeros([12])
        self.amp_bc = 0.3
        self.amp_ft = 0.188
        #self.amp_bc = 0.3
        #self.amp_ft = 0.188
        self.old_z = np.zeros(12)
        self.fre = 0
        self.theta1 = [[0, 0, 0], [1 / 4, 0, 0], [1 / 4, 0, 0]]
        self.theta1 = [[0, 0, 0], [1 / 6+1/72, 0, 0], [1 / 6+1/72, 0, 0]]
        self.theta1 = [[0, 0, 0], [1 / 12+1/36, 0, 0], [1 / 12+1/36, 0, 0]]
        #self.cf_base = 0.35  # 右边的零位状态　左边增加负号
        #self.ft_base = -0.35
        self.cf_base = 0.3  # 右边的零位状态　左边增加负号
        self.ft_base = -0.33
        self.old_zz = np.zeros((6,6))

        for i in range(6):
            for j in range(6):
                self.theta[i][j] = self.theta[i][j] * 2 * np.pi

        for i in range(3):
            for j in range(3):
                self.theta1[i][j] = self.theta1[i][j] * 2 * np.pi
        # 仅仅针对三足步态
        with open('initial.json', 'r') as f:
            data_read = json.load(f)
            self.theta0 = np.asarray(data_read['joint_angles'])
            self.zz0 = np.asarray(data_read['CPG_output'])


    def hopf6_3(self, z, t):
        for i in range(6):
            #w = self.w_stance / (np.exp(-self.a * z[2 * i + 1]) + 1) + self.w_swing / (
                        #np.exp(self.a * z[2 * i + 1]) + 1)
            w=self.w_a/2/self.b+(2*self.b-1)*self.w_a/(2*self.b*(1-self.b)*(1+(np.exp(-self.a * z[2 * i + 1]))))
            self.dzdt[2 * i] = self.p * (self.u - pow(z[2 * i + 1], 2) - pow(z[2 * i], 2)) * z[2 * i] - w * z[2 * i + 1]
            self.dzdt[2 * i + 1] = self.p * (self.u - pow(z[2 * i + 1], 2) - pow(z[2 * i], 2)) * z[2 * i + 1] + w * z[
                2 * i]
            for j in {0}:
                if j<0:
                    j=j+6
                elif j>5:
                    j=j-6
                self.dzdt[2 * i] += self.k1 * (
                            np.cos(self.theta[i][j]) * z[2 * j] - np.sin(self.theta[i][j]) * z[2 * j + 1])
                self.dzdt[2 * i + 1] += self.k1 * (
                            np.sin(self.theta[i][j]) * z[2 * j] + np.cos(self.theta[i][j]) * z[2 * j + 1])
        return self.dzdt

    def cpg_output(self, z0, t0, tf):
        t_eval = np.arange(t0, tf, self.dt)
        sol = odeint(self.hopf6_3, z0, t_eval)
        return sol[-1, :]

    def sec_hopf(self, z1):
        dz1 = np.zeros([6, 6])
        for j in range(6):  # 腿编号
            for i in [1, 2]:
                #w = self.w_stance / (np.exp(-self.a * z1[j, 2 * i + 1]) + 1) + self.w_swing / (
                            #np.exp(self.a * z1[j, 2 * i + 1]) + 1)
                w=self.w_a/2/self.b+(2*self.b-1)*self.w_a/(2*self.b*(1-self.b)*(1+(np.exp(-self.a * z1[j,2 * i + 1]))))
                dz1[j, 2 * i] = self.p * (self.u - pow(z1[j, 2 * i + 1], 2) - pow(z1[j, 2 * i], 2)) * z1[j, 2 * i] - w * \
                                z1[j, 2 * i + 1]
                dz1[j, 2 * i + 1] = self.p * (self.u - pow(z1[j, 2 * i + 1], 2) - pow(z1[j, 2 * i], 2)) * z1[
                    j, 2 * i + 1] + w *z1[j, 2 * i]
                for k in range(3):
                    dz1[j, 2 * i] += self.k1 * (
                                np.cos(self.theta1[i][k]) * z1[j, 2 * k] - np.sin(self.theta1[i][k]) * z1[j, 2 * k + 1])
                    dz1[j, 2 * i + 1] += self.k1 * (
                            np.sin(self.theta1[i][k]) * z1[j, 2 * k] + np.cos(self.theta1[i][k]) * z1[j, 2 * k + 1])
        return dz1

    def two_layer_out(self, z1, t0, tf):
        z0=np.zeros(12)
        for j in range(6):
            z0[2*j:2*j+2]=z1[j,0:2]
        t_eval = np.arange(t0, tf, self.dt)
        sol = odeint(self.hopf6_3, z0, t_eval)
        # sol2=odeint(self.sec_hopf,z1,t_eval)
        count = 0
        zz_out = z1
        zz_out = np.resize(zz_out, (6, 6, 1))
        for t in np.arange(t0, tf - self.dt, self.dt):
            dz1 = self.sec_hopf(z1)
            count += 1
            for j in range(6):
                z1[j, 0:2] = sol[count, j * 2:2 * j + 2]
            z1 = z1 + dz1 * self.dt
            zz_out = np.append(zz_out, np.resize(z1, (6, 6, 1)), axis=2)

        return zz_out[:, :, -1]

    def two_layer_out_all(self, z1, t0, tf):
        z0=np.zeros(12)
        for j in range(6):
            z0[2*j:2*j+2]=z1[j,0:2]
        t_eval = np.arange(t0, tf, self.dt)
        sol = odeint(self.hopf6_3, z0, t_eval)
        # sol2=odeint(self.sec_hopf,z1,t_eval)
        count = 0
        zz_out = z1
        zz_out = np.resize(zz_out, (6, 6, 1))
        for t in np.arange(t0, tf - self.dt, self.dt):
            dz1 = self.sec_hopf(z1)
            count += 1
            for j in range(6):
                z1[j, 0:2] = sol[count, j * 2:2 * j + 2]
            z1 = z1 + dz1 * self.dt
            zz_out = np.append(zz_out, np.resize(z1, (6, 6, 1)), axis=2)

        return zz_out

    def cpg_begin(self, z0):
        z0 = np.ones(12) * 0.5
        z1 = np.ones((6, 6)) * 0.5
        current_T = 0
        cpg_o = np.ones([1, 12]) * 0.5
        T = [0]
        n = 2

        z0 = self.cpg_output(z0, current_T, current_T + 240 * n * self.dt)
        current_T += 240 * n * self.dt
        while current_T < 10:

            z0 = self.cpg_output(z0, current_T, current_T + (n + 1) * self.dt)
            current_T += n * self.dt

            z1 = np.resize(z0, (1, 12))

            if current_T > 4 and abs(z0[2]) < 0.001:
                print("find")
                break
            if current_T > 2:
                cpg_o = np.append(cpg_o, z1, axis=0)
                T.append(current_T)
        print(z0)

        '''
        fre=np.zeros(12)
        for j in range(12):
            fre[j] = freq_from_crossings(cpg_o[2:, j], 1.0 / (n * self.dt))
        fre2=freq_from_fft(cpg_o[2:, 0], 1.0 / (n * self.dt))
        fre3=freq_from_autocorr(cpg_o[2:, 0], 1.0 / (n * self.dt))
        plt.figure()
        for j in range(6):

            plt.subplot(6,1,j+1)
            plt.plot(T, cpg_o[:, 2 * j])
        plt.show()
        self.fre=sum(fre)/12.0
        print(fre)
        print(self.fre)
        '''
        return z0

    def cpg_begin_all(self, z0):
        z00 = np.ones(12) * 0.5
        z2 = np.ones((6, 6)) * 0.5
        current_T = 0
        cpg_o = np.ones([1, 12]) * 0.5
        T = [0]
        n = 2

        # z0 = self.cpg_output(z00, current_T, current_T + 240*n * self.dt)
        # current_T +=240*n*self.dt
        while current_T < 10:

            z0 = self.cpg_output(z0, current_T, current_T + (0.8 + n) * self.dt)
            # z3 = self.cpg_output(z00, 0, current_T + (0.8+n)*self.dt)
            current_T += n * self.dt

            z1 = np.resize(z0, (1, 12))

            if current_T > 3 and abs(z0[2]) < 0.01:
                print("find")
                break
                '''
            if current_T > 2:
                cpg_o = np.append(cpg_o, z1, axis=0)
                T.append(current_T)
                '''
        print(z0)
        cpg_o1 = self.two_layer_out(z2, 0, current_T)
        # k11=self.cpg_output(z00,0,current_T)

        '''
        fre=np.zeros(12)
        for j in range(12):
            fre[j] = freq_from_crossings(cpg_o[2:, j], 1.0 / (n * self.dt))
        fre2=freq_from_fft(cpg_o[2:, 0], 1.0 / (n * self.dt))
        fre3=freq_from_autocorr(cpg_o[2:, 0], 1.0 / (n * self.dt))
        plt.figure()
        for j in range(6):

            plt.subplot(6,1,j+1)
            plt.plot(T, cpg_o[:, 2 * j])
        plt.show()
        self.fre=sum(fre)/12.0
        print(fre)
        print(self.fre)
        '''
        return cpg_o1

    def bc_cpg2theta(self, z):
        bc_theta = np.zeros(6)
        for j in range(6):
            bc_theta[j] = self.amp_bc * z[2 * j]
        return bc_theta

    def bc_theta2cpg(self, bc_theta, z0):
        z = np.zeros(12)
        for j in range(6):
            z[2 * j] = bc_theta[j] / self.amp_bc
            z[2 * j + 1] = z0[2 * j + 1]
        return z

    def diff_cpg(self, z1):
        z0 = self.old_z
        dz = z1 - z0
        if dz > 0:
            dz = 1
        else:
            dz = 0
        return dz

    def cpg2theta(self, zz):
        theta = np.zeros((6, 3))
        for j in range(6):
            theta[j, 0] = self.amp_bc * zz[j, 0]
            if j >= 3:
                theta[j, 0] = -self.amp_bc * zz[j, 0]
            if j < 3:
                theta[j, 1] = -self.cf_base + self.amp_ft * zz[j, 2]
                theta[j, 2] = -self.ft_base + self.amp_ft * zz[j, 4]
            else:
                theta[j, 1] = self.cf_base - self.amp_ft * zz[j, 2]  # cf
                theta[j, 2] = self.ft_base - self.amp_ft * zz[j, 4]  # ft

        return theta

    def theta2cpg(self, theta, zz0):
        zz = np.zeros((6, 6))
        for j in range(6):
            zz[j, 0] = theta[j, 0] / self.amp_bc
            zz[j, 1] = zz0[j, 1]
            if j == 3:
                zz[j, 0] = -theta[j, 0] / self.amp_bc
            if j < 3:
                zz[j, 2] = (theta[j, 1] + self.cf_base) / self.amp_ft
                zz[j, 3] = zz0[j, 3]
                zz[j, 4] = (theta[j, 2] + self.ft_base) / self.amp_ft
                zz[j, 5] = zz0[j, 5]
            else:
                zz[j, 2] = (-theta[j, 1] + self.cf_base) / self.amp_ft
                zz[j, 3] = zz0[j, 3]
                zz[j, 4] = (-theta[j, 2] + self.ft_base) / self.amp_ft
                zz[j, 5] = zz0[j, 5]

        return zz

    def cpg2theta_reshape(self, zz):
        theta = np.zeros((6, 3))
        bound0=0
        for j in range(6):
            theta[j, 0] = self.amp_bc * zz[j, 0]
            if j >=3:
                theta[j, 0] = -self.amp_bc * zz[j, 0]
            if j < 3:
                if zz[j, 2] <bound0:
                    theta[j, 1] = -self.cf_base + self.amp_ft * 0
                else:
                    theta[j, 1] = -self.cf_base + self.amp_ft * zz[j, 2] # 伸展为加
                if zz[j, 4]<bound0:
                    theta[j, 2] = -self.ft_base + self.amp_ft * 0
                else:
                    theta[j, 2] = -self.ft_base + self.amp_ft * zz[j, 4]

            else:
                if zz[j, 2] <bound0:
                    theta[j, 1] = self.cf_base - self.amp_ft * 0  # cf
                else:
                    theta[j, 1] = self.cf_base - self.amp_ft * zz[j, 2]  # cf
                if zz[j, 4]<bound0:
                    theta[j, 2] = self.ft_base - self.amp_ft * 0  # ft
                else:
                    theta[j, 2] = self.ft_base - self.amp_ft * zz[j, 4]  # ft


        return theta
    

    def cpg2theta_reshape1(self, zz):
        theta = np.zeros((6, 3))
        bound0=-0.01
        for j in range(6):
            theta[j, 0] = self.amp_bc * zz[j, 0]
            if j >=3:
                theta[j, 0] = -self.amp_bc * zz[j, 0]
            if j < 3:
                if zz[j, 2] <bound0:
                    theta[j, 1] = -self.cf_base + self.amp_ft * bound0
                else:
                    theta[j, 1] = -self.cf_base + self.amp_ft * zz[j, 2] # 伸展为加
                if zz[j, 4]<bound0:
                    theta[j, 2] = -self.ft_base + self.amp_ft * bound0
                else:
                    theta[j, 2] = -self.ft_base + self.amp_ft * zz[j, 4]

            else:
                if zz[j, 2] <bound0:
                    theta[j, 1] = self.cf_base - self.amp_ft * (bound0)  # cf
                else:
                    theta[j, 1] = self.cf_base - self.amp_ft * zz[j, 2]  # cf
                if zz[j, 4]<bound0:
                    theta[j, 2] = self.ft_base - self.amp_ft * (bound0)  # ft
                else:
                    theta[j, 2] = self.ft_base - self.amp_ft * zz[j, 4]  # ft


        return theta
    
    def cpg2theta_reshape2(self, zz):
        theta = np.zeros((6, 3))
        bound0=0.7
        bound0=0.88
        coef=0.05
        for j in range(6):
            theta[j, 0] = self.amp_bc * zz[j, 0]
            if j >=3:
                theta[j, 0] = -self.amp_bc * zz[j, 0]
            if j < 3:
                if zz[j, 2] <bound0:
                    theta[j, 1] = -self.cf_base + self.amp_ft * zz[j, 2]*coef
                else:
                    theta[j, 1] = -self.cf_base + self.amp_ft * bound0*coef+self.amp_ft * (zz[j, 2]-bound0)*1/(1-bound0) # 伸展为加
                if zz[j, 4]<bound0:
                    theta[j, 2] = -self.ft_base + self.amp_ft * zz[j, 4]*coef
                else:
                    theta[j, 2] = -self.ft_base + self.amp_ft * bound0*coef+ self.amp_ft * (zz[j, 4]-bound0)*1/(1-bound0)

            else:
                if zz[j, 2] <bound0:
                    theta[j, 1] = self.cf_base - self.amp_ft * zz[j, 2]*coef  # cf
                else:
                    theta[j, 1] = self.cf_base - self.amp_ft * bound0*coef-self.amp_ft * (zz[j, 2]-bound0)*1/(1-bound0)  # cf
                if zz[j, 4]<bound0:
                    theta[j, 2] = self.ft_base - self.amp_ft * zz[j, 4]*coef  # ft
                else:
                    theta[j, 2] = self.ft_base - self.amp_ft * bound0*coef- self.amp_ft * (zz[j, 4]-bound0)*1/(1-bound0)  # ft


        return theta

    def get_phase(self,zz):
        phase=np.zeros(6)
        bound0=0
        for j in range(6):
            if j ==0 or j==3:
                bound0=0
            if j==1 or j ==4:
                bound0=0.
            if j==2 or j ==5:
                bound0=0
            if j < 6:
                if zz[j, 2] <bound0:
                    phase[j] = -1 # 支撑相
                else:
                    if zz[j,2]==bound0:
                        phase[j] = 0
                    else:
                        phase[j] = 1

        return  phase

    def add_reflex(self,theta):
        return 1




class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
