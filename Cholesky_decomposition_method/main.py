import numpy as np

from src.util.Cholesky import CholeskyDecomposition
from src.util.multivarivate_simulator import MultivariateNormalSample, PriceSimulator
from src.util.maxrainbowpayoff import MaxRainbowPayoff

K = 100
r = 0.1
T = 0.5
N = 10000
rep = 20
n = 2

S0 = np.zeros((1, n))
q = np.zeros((1, n))
sigma = np.zeros((1, n))
rho = np.zeros((n, n))

# input S0
for j in range(n):
    S0[0, j] = float(input(f"S{j+1}0 = "))
# input q
for j in range(n):
    q[0, j] = float(input(f"q{j+1} = "))
# input sigma
for j in range(n):
    sigma[0, j] = float(input(f"sigma{j+1} = "))
# input rho
for j in range(n):
    for i in range(n):
        if i == j:
            rho[i, j] = 1
        elif i > j:
            rho[i, j] = float(input(f"rho{i+1}{j+1} = "))
        else:
             rho[i, j]= rho[j, i]

# get covariance matrix
covariance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            covariance_matrix[i, j] = sigma[0, i] ** 2
        elif i < j:
            covariance_matrix[i, j] = rho[i, j] * sigma[0, i] * sigma[0, j]
        else:
            covariance_matrix[i, j] = covariance_matrix[j, i]


# get triangular matrix
Cholesky = CholeskyDecomposition(n, covariance_matrix * T) # remember that T !!!
triangular_matrix = Cholesky.get_triangular_matrix()

#######################################
# Basic Requirement
#######################################
values_ls = []
for i in range(rep):

    # get basic normal sample
    get_normal_sample = MultivariateNormalSample(N, n)
    normal_sample = get_normal_sample.basic_normal_sample()

    # get simulated prices
    price_simulator = PriceSimulator(N, n, S0, r, q, sigma, T)
    price_matrix = price_simulator.price_simulate(normal_sample, triangular_matrix)

    # get payoff
    payoff = MaxRainbowPayoff(K)
    payoff_matrix = payoff.get_payoff_matrix(price_matrix)

    # get option values by discounted payoff
    value = np.exp(-r * T) * np.mean(payoff_matrix)
    values_ls.append(value)

values_arr = np.array(values_ls)
CI1 = values_arr.mean() - 2 * values_arr.std()
CI2 = values_arr.mean() + 2 * values_arr.std()

print("Basic Cholesky Decomposition Method")
print(f"maximum rainbow call value = {values_arr.mean()}")
print(f"95% C.I.:[{CI1}, {CI2}]")
print(f"variance: {values_arr.std()}")

##############
# Bonus 1 + 2
##############
anti_mom_values_ls = []
inv_values_ls = []
for i in range(rep):

    # get antithetic normal sample
    # bonus 1 和 bonus 2 要用同一組 antithetic normal sample 才能比較其誤差大小
    get_normal_sample = MultivariateNormalSample(N, n)
    anti_normal_sample = get_normal_sample.anti_normal_sample()

    # 建立物件
    price_simulator = PriceSimulator(N, n, S0, r, q, sigma, T)
    payoff = MaxRainbowPayoff(K)

    ### Bonus 1 : antithetic variate + moment matching ###
    anti_mom_normal_sample = get_normal_sample.anti_mom_normal_sample(anti_normal_sample)
    # get simulated prices
    anti_mom_price_matrix = price_simulator.price_simulate(anti_mom_normal_sample, triangular_matrix)
    # get payoff
    anti_mom_payoff_matrix = payoff.get_payoff_matrix(anti_mom_price_matrix)
    # get option values by discounted payoff
    anti_mom_value = np.exp(-r * T) * np.mean(anti_mom_payoff_matrix)
    anti_mom_values_ls.append(anti_mom_value)


    ### Bonus 2 : inverse Cholesky method ###
    inv_normal_sample = get_normal_sample.inverse_Cholesky(anti_normal_sample)
    # get simulated prices
    inv_price_matrix = price_simulator.price_simulate(inv_normal_sample, triangular_matrix)
    # get payoff
    inv_payoff_matrix = payoff.get_payoff_matrix(inv_price_matrix)
    # get option values by discounted payoff
    inv_value = np.exp(-r * T) * np.mean(inv_payoff_matrix)
    inv_values_ls.append(inv_value)


### Bonus 1 ###
anti_mom_values_arr = np.array(anti_mom_values_ls)
CI1 = anti_mom_values_arr.mean() - 2 * anti_mom_values_arr.std()
CI2 = anti_mom_values_arr.mean() + 2 * anti_mom_values_arr.std()

print("\nBonus 1: Antithetic variate + Moment matching Method")
print(f"maximum rainbow call value = {anti_mom_values_arr.mean():.4f}")
print(f"95% C.I.:[{CI1:.4f}, {CI2:.4f}]")
print(f"Interval length = {(CI2 - CI1):.6f} ")


### Bonus 2 ###
inv_values_arr = np.array(inv_values_ls)
CI1 = inv_values_arr.mean() - 2 * inv_values_arr.std()
CI2 = inv_values_arr.mean() + 2 * inv_values_arr.std()

print("\nBonus 2: Inverse Cholesky Decomposition Method")
print(f"maximum rainbow call value = {inv_values_arr.mean():.4f}")
print(f"95% C.I.:[{CI1:.4f}, {CI2:.4f}]")
print(f"Interval length = {(CI2 - CI1):.6f} ")





#%%
##########################################################
# Variance Reduction: antithetic variate + moment matching
##########################################################
values_ls = []
for i in range(rep):
    # get basic normal sample
    get_normal_sample = MultivariateNormalSample(N, n)
    normal_sample = get_normal_sample.anti_mom_normal_sample()

    # get simulated prices
    price_simulator = PriceSimulator(N, n, S0, r, q, sigma, T)
    price_matrix = price_simulator.price_simulate(normal_sample, triangular_matrix)

    # get payoff
    payoff = MaxRainbowPayoff(K)
    payoff_matrix = payoff.get_payoff_matrix(price_matrix)

    # get option values by discounted payoff
    value = np.exp(-r * T) * np.mean(payoff_matrix)
    values_ls.append(value)

values_arr = np.array(values_ls)
CI1 = values_arr.mean() - 2 * values_arr.std()
CI2 = values_arr.mean() + 2 * values_arr.std()

print("\nAntithetic variate + Moment matching Method")
print(f"maximum rainbow call value = {values_arr.mean()}")
print(f"95% C.I.:[{CI1}, {CI2}]")
print(f"variance: {values_arr.std()}")

#############################################
# Variance Reduction: inverse Cholesky method
#############################################
values_ls = []
for i in range(rep):
    # get triangular matrix
    Cholesky = CholeskyDecomposition(n, covariance_matrix * T) # remember that T !!!
    triangular_matrix = Cholesky.get_triangular_matrix()

    # get basic normal sample
    get_normal_sample = MultivariateNormalSample(N, n)
    normal_sample = get_normal_sample.inverse_Cholesky()

    # get simulated prices
    price_simulator = PriceSimulator(N, n, S0, r, q, sigma, T)
    price_matrix = price_simulator.price_simulate(normal_sample, triangular_matrix)

    # get payoff
    payoff = MaxRainbowPayoff(K)
    payoff_matrix = payoff.get_payoff_matrix(price_matrix)

    # get option values by discounted payoff
    value = np.exp(-r * T) * np.mean(payoff_matrix)
    values_ls.append(value)

values_arr = np.array(values_ls)
CI1 = values_arr.mean() - 2 * values_arr.std()
CI2 = values_arr.mean() + 2 * values_arr.std()

print("\nInverse Cholesky Decomposition Method")
print(f"maximum rainbow call value = {values_arr.mean()}")
print(f"95% C.I.:[{CI1}, {CI2}]")
print(f"variance: {values_arr.std()}")