import sympy as sym


A = sym.Matrix([[1, 2], [3, 4]])
B = sym.Matrix([[5, 6], [7, 8]])


innerp = 0
for i in range(2):
    for j in range(2):
        innerp += A[i, j] * B[i, j]

print(sym.MatMul(A, B))
