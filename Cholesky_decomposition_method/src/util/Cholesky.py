import numpy as np
from math import sqrt

class CholeskyDecomposition:
    def __init__(self, n, covariance_matrix):
        self.n = n
        self.covariance_matrix = covariance_matrix

    def get_triangular_matrix (self):

        triangular_matrix = np.zeros((self.n, self.n))
        triangular_matrix[0, 0] = sqrt(self.covariance_matrix[0,0])

        for j in range(1, self.n):
            triangular_matrix[0, j] = self.covariance_matrix[0, j] / triangular_matrix[0, 0]

        for i in range(1, self.n - 1):
            # for the main diagonal
            sum1 = 0
            for k in range(i):
                sum1 += pow(triangular_matrix[k, i], 2)
            triangular_matrix[i, i] = sqrt(self.covariance_matrix[i, i] - sum1)

            # for the rest at right of the main diagonal for each row
            for j in range(i + 1, self.n):
                sum2 = 0
                for k in range(i):
                    sum2 += triangular_matrix[k, i] * triangular_matrix[k, j]
                triangular_matrix[i, j] = (self.covariance_matrix[i, j] - sum2) / triangular_matrix[i, i]


        sum3 = 0
        for k in range(self.n - 1):
            sum3 += pow(triangular_matrix[k, self.n - 1], 2)
        triangular_matrix[self.n -1, self.n - 1] = sqrt(self.covariance_matrix[self.n - 1, self.n - 1] - sum3)

        return triangular_matrix


# for testing
if __name__ == "__main__":
    from scipy import linalg
    mat = np.eye(100)
    mat_A = CholeskyDecomposition(100, mat)
    A = mat_A.get_triangular_matrix()
    print(A)
    print(linalg.cholesky(mat))


## for testing
if __name__ == "__main__":
    from scipy import linalg
    # n = 4
    # covariance_matrix = np.array([[16,4,4,-4],[4,10,4,2],[4,4,6,-2],[-4,2,-2,4]])
    n = 2
    covariance_matrix = np.array([[0.25, 0.25*0.5],[0.25*0.5,0.25]])

    Cholesky = CholeskyDecomposition(n, covariance_matrix)
    triangular_matrix = Cholesky.get_triangular_matrix()
    tri = linalg.cholesky(triangular_matrix)
    print(triangular_matrix)
    print(triangular_matrix.T @ triangular_matrix) # 矩陣乘法
    print(tri.T @ tri)


# #%%
# # Cholesky decomposition
# import numpy as np
# from math import sqrt, pow
#
# n = 4
# covariance_matrix = np.array([[16,4,4,-4],[4,10,4,2],[4,4,6,-2],[-4,2,-2,4]])
#
# A = np.zeros((n, n))
# A[0, 0] = sqrt(covariance_matrix[0,0])
# for j in range(1, n):
#     A[0, j] = covariance_matrix[0, j] / A[0, 0]
#
# for i in range(1, n - 1):
#     for j in range(i, n):
#         sum1 = 0
#         sum2 = 0
#         for k in range(i):
#             sum1 += pow(A[k, i], 2)
#             sum2 += A[k, i] * A[k, j]
#             if i == j:
#                 A[i, i] = sqrt(covariance_matrix[i, i] - sum1)
#             elif i < j:
#                 A[i, j] = (covariance_matrix[i, j] - sum2) / A[i, i]
#             else:
#                 A[i, j] = 0
#
# sum3 = 0
# for k in range(n - 1):
#     sum3 += pow(A[k,n - 1], 2)
# A[n -1, n - 1] = sqrt(covariance_matrix[n - 1, n -1] - sum3)
#
#
# print(covariance_matrix)
# print(A)
# print(A.T @ A)  # 矩陣乘法
#
# #%%
# # test sum
# import numpy as np
# from math import sqrt, pow
#
# C = np.array([[1,2,3]])
# print(pow(C[0,1],2))
# print(C)
# sum_ls = []
# for i in range(3):
#     # 3,1 = arr.shape
#     sum_ls.append([])
#     sum_ls[i] = pow(C[0, i], 2)
#     # A[0, i] = sum
#
# print(sum(sum_ls))
#
# #%%
# # test rho matrix
# import numpy as np
# n = 3
# rho = np.zeros((n, n))
# for j in range(n):
#     for i in range(n):
#         if i == j:
#             rho[i, j] = 1
#         elif i > j:
#             rho[i, j] = float(input(f"rho{i+1}{j+1} = "))
#         else:
#              rho[i, j]= rho[j, i]
# print(rho)
#
# #%%
# for i in range(3,4):
#     print(i)