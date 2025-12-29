import numpy as np

np.set_printoptions(suppress=True, linewidth=200)

Data_p = [-18,99,-162]
Data_q = [-15,54]


def Matrix_Companion_p(Data):
    n = len(Data)
    A = np.zeros((n, n))
    A[0, :] = -np.array(Data)
    for i in range(1, n):
        A[i, i - 1] = 1
    return A

def Matrix_Companion_q(Data):
    m = len(Data)
    A = np.zeros((m, m))
    A[0, :] = -np.array(Data)
    for i in range(1, m):
        A[i, i - 1] = 1
    return A

A = Matrix_Companion_p(Data_p)
print("Ma trận đồng hành của p:\n", A)
print("_" * 100)

B = Matrix_Companion_q(Data_q)
print("Ma trận đồng hành của q:\n", B)
print("_" * 100)
m = A.shape[0] 
n = B.shape[0] 
I_m = np.eye(m) 
I_n = np.eye(n) 
Tong_Kron = np.kron(A, I_n) + np.kron(I_m, -B)
print("Tổng Kronecker:")
print(Tong_Kron)
w, v = np.linalg.eig(Tong_Kron)
print(".......................")
print("1. Các trị riêng (w):", w)
print("2. Ma trận chứa vector riêng (v):\n", v)
determinant_Tong_Kron = np.linalg.det(Tong_Kron)
print("DInh thuc = ",determinant_Tong_Kron )
print("1 = ",np.kron(A, I_n))