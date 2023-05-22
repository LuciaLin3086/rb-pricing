import numpy as np
import pandas as pd
from math import log, pow
from scipy.linalg import inv

from .Cholesky import CholeskyDecomposition

class MultivariateNormalSample:
    def __init__(self, N, n):
        self.N = N
        self.n = n

    def basic_normal_sample(self):
        normal_sample = np.random.randn(self.N, self.n)  # std normal 抽樣出 Z1,Z2,...,Zn
        return normal_sample

    def anti_normal_sample(self): # antithetic variate
        anti_normal_ls = []
        for j in range(self.n): # bonus 1
            normal_sample = np.random.randn(self.N)
            for i in range(int(self.N / 2)):
                normal_sample[i + int(self.N / 2)] = - normal_sample[i]

            anti_normal_ls.append(normal_sample)

        anti_normal_sample = np.array(anti_normal_ls).T  # make Z matrix be N * n

        return anti_normal_sample

    def anti_mom_normal_sample(self, anti_normal_sample): # bonus 1: antithetic variate + moment matching
        normal_sample = np.zeros((self.N, self.n))
        for j in range(self.n):
            std = anti_normal_sample[:, j].std()
            for i in range(self.N):
                normal_sample[i, j] = anti_normal_sample[i, j] / std

        return normal_sample


    def inverse_Cholesky(self, anti_normal_sample): # bonus 2
        basic_normal_sample = pd.DataFrame(anti_normal_sample)

        covariance_matrix = basic_normal_sample.cov().values
        Cholesky = CholeskyDecomposition(self.n, covariance_matrix)
        triangular_matrix = Cholesky.get_triangular_matrix()
        inverse_triangular_matrix = inv(triangular_matrix)

        # pandas 是 numpy 再包裝，所以用 .values 將pandas轉為numpy
        normal_sample = basic_normal_sample.values @ inverse_triangular_matrix

        return normal_sample


    # def anti_mom_normal_sample(self):
    #     normal_ls = []
    #     for j in range(self.n):
    #         anti_normal_sample = np.random.randn(self.N)
    #         for i in range(int(self.N / 2)):
    #             anti_normal_sample[i + int(self.N / 2)] = - anti_normal_sample[i]  # antithetic variate
    #
    #         one_normal_sample = anti_normal_sample / anti_normal_sample.std()  # moment matching
    #         normal_ls.append(one_normal_sample)
    #
    #     normal_sample = np.array(normal_ls).T  # make Z matrix be N * n
    #     return normal_sample
    #
    # def inverse_Cholesky(self):
    #     basic_normal_sample = pd.DataFrame(np.random.randn(self.N, self.n))
    #     for i in range(self.n):
    #         # basic_normal_sample[i] : i-th column (by pandas)
    #         basic_normal_sample[i] = basic_normal_sample[i] - basic_normal_sample.mean(axis=0)[i] # axis=0 縱向運算
    #
    #     # get inverse Cholesky
    #     covariance_matrix = basic_normal_sample.cov().values
    #     Cholesky = CholeskyDecomposition(self.n, covariance_matrix)
    #     triangular_matrix = Cholesky.get_triangular_matrix()
    #     inverse_triangular_matrix = inv(triangular_matrix)
    #
    #     # pandas 是 numpy 再包裝，所以用 .values 將pandas轉為numpy
    #     normal_sample = basic_normal_sample.values @ inverse_triangular_matrix
    #
    #     return normal_sample


class PriceSimulator:
    def __init__(self, N, n, S0, r, q, sigma, T):
        self.N = N
        self.n = n
        self.S0 = S0
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T

    def price_simulate(self, normal_sample, triangular_matrix):
        multivarivate_sample = normal_sample @ triangular_matrix # r1,r2,...,rn

        lnS = np.zeros((self.N, self.n))
        for i in range(self.N):
            for j in range(self.n):
                lnS[i, j] = multivarivate_sample[i, j] + log(self.S0[0, j]) + \
                            (self.r - self.q[0, j] - pow(self.sigma[0, j], 2) / 2) * self.T # ln(ST)分配

        return np.exp(lnS)  # 抽樣後的stock price # 因為是矩陣，所以用np，而不是math





#######################################
# Basic Requirement
#######################################
# class Price_simulator:
#     def __init__(self, N, n, S0, r, q, sigma, T):
#         self.N = N
#         self.n = n
#         self.S0 = S0
#         self.r = r
#         self.q = q
#         self.sigma = sigma
#         self.T = T
#
#     def price_simulate(self, triangular_matrix):
#         normal_sample = np.random.randn(self.N, self.n)  # std normal 抽樣出 Z1,Z2,...,Zn
#
#         multivarivate_sample = normal_sample @ triangular_matrix # r1,r2,...,rn
#
#         lnS = np.zeros((self.N, self.n))
#         for i in range(self.N):
#             for j in range(self.n):
#                 lnS[i, j] = multivarivate_sample[i, j] + log(self.S0[0, j]) + \
#                             (self.r - self.q[0, j] - pow(self.sigma[0, j], 2) / 2) * self.T # ln(ST)分配
#
#         return np.exp(lnS)  # 抽樣後的stock price # 因為是矩陣，所以用np，而不是math
#
#



## for testing
if __name__ == "__main__":
    N = 10000
    n = 3
    S0 = np.array([[100, 110, 120]])
    r = 0.05
    q = np.array([[0.01, 0.02, 0.03]])
    sigma = np.array([[0.4, 0.5, 0.6]])
    T = 0.6
    triangular_matrix = np.array([[2,1,3], [0,2,2], [0,0,1]])


    price_simulator = PriceSimulator(N, n, S0, r, q, sigma, T)
    price_matrix = price_simulator.price_simulate(triangular_matrix)
    print(price_matrix)


#%%
# for testing
# n = 3
# N = 4
#
# normal_ls = []
# for j in range(n):
#     normal_sample = np.random.randn(N)
#
#     for i in range(int(N / 2)):
#         normal_sample[i + int(N / 2)] = - normal_sample[i]
#
#     one_normal_sample = normal_sample / normal_sample.std()
#     normal_ls.append(one_normal_sample)
#
# var_red_normal_sample = np.array(normal_ls)
# print(var_red_normal_sample.T)
