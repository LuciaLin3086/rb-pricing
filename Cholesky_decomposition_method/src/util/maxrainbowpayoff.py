import numpy as np

# Calculate the option payoff of the simulated prices.
class MaxRainbowPayoff:
    def __init__(self, K):
        self.K = K

    # def get_max_price(self, N, n, price_matrix):
    #     max_S = np.zeros((N, 1))
    #     for i in range(N):
    #         for j in range(n):
    #             max_S[i, 0] = max(max_S[i, 0], price_matrix[i, j])
    #
    #     return max_S

    # def get_payoff_matrix(self, N, n, price_matrix):
    #
    #     payoff_matrix = np.zeros((N, 1))
    #     for i in range(N):
    #         payoff_matrix[i, 0] = max(self.get_max_price(N, n, price_matrix)[i, 0] - self.K, 0)

    def get_payoff_matrix(self, price_matrix):
        payoff_ls = []
        max_S = np.max(price_matrix, axis = 1) # axis = 1 代表橫向運算
        payoff = np.where(max_S - self.K > 0, max_S - self.K, 0)
        payoff_ls.append(payoff)

        return np.array(payoff_ls)

## for testing
if __name__ == "__main__":
    N = 4
    n = 3
    price_matrix = np.array([[100,110,120], [102,134,128],[90,80,70],[105,112,113]])
    K = 114

    payoff = MaxRainbowPayoff(K)
    payoff_matrix = payoff.get_payoff_matrix(price_matrix)
    print(payoff_matrix)