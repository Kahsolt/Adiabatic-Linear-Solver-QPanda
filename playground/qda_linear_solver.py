#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/15

# 第二问: 线性方程组问题 A * x = b 转 绝热演化问题 H(s) = (1 - f(s)) * H0 + f(s) * H1
# arXiv:1909.05500
# arXiv:2111.08152 Chap. IV

from code import interact
from scipy.linalg import expm
from utils import *


# classical solution
print(f'A: (norm={np.linalg.norm(Am):.5f})')
print(Am)
print(f'b: (norm={np.linalg.norm(bv):.5f})')
print(bv)
x = np.linalg.solve(Am, bv)
print('x:', x.T[0])

print('=' * 42)

print(f'A: (norm={np.linalg.norm(A):.5f})')
print(A)
print(f'b: (norm={np.linalg.norm(b):.5f})')
print(b)
x = np.linalg.solve(A, b)
print('x:', x.T[0])

# arXiv:1909.05500, vanilla AQC
Qb = np.eye(N) - b @ b.conj().T
H0 = np.kron(σx, Qb)
H1 = np.kron(σp, A @ Qb) + np.kron(σm, Qb @ A)
print('H0')
print(H0)
print('H1')
print(H1)

# https://byjus.com/question-answer/what-is-nullspace-of-a-matrix/
H0_nullspace = [    # H0 @ null(H0) = 0
  np.kron(v0, b),   # \tilde{b}
  np.kron(v1, b),   # \bar{b}
]
H1_nullspace = [    # H1 @ null(H1) = 0
  np.kron(v0, x),   # \tilde{x}; if A|x> ~ |b>, then Qb*A|x> = Qb*|b> = 0
  np.kron(v1, b),   # \bar{b}
]

# Let f: [0, 1] -> [0, 1] be a scheduling function, we'll have the AQC evolution:
#   H(f(s)) = (1 - f(s)) * H0 + f(s) * H1
# and the null-space for H(f(s)) is:
#   H1_nullspace = [
#     np.kron(v0, x(s)),   # \tilde{x}(s)
#     np.kron(v1, b),      # \bar{b}
#   ]
# where |\tilde{x}(s=0)> = |\tilde{b}> and |\tilde{x}(s=1)> = |\tilde{x}> for **any** s
# hence |\tilde{x}(s)> is the **desired adiabatic path**
# Let P0(s) be the projection to the subspace of Null(H(f(s))):
#   P0(s) = |\tilde{x}(s)><\tilde{x}(s)| + |\bar{b}><\bar{b}|

# 如何做第二问：
#  1. 制备初态: |ψ_T(0)> = |\tilde{b}>
#  2. 构建线路进行 200 步含时演化 (虚时演化/QAOA/量子随机游走/...各种方案)
#  3. 读出末态: |ψ_T(1)> = |\tilde{x}>, 解出 |x>

# 制备初态
qs = H0_nullspace[0]
print('qstate init:', qs)
# 绝热演化 arXiv:1909.05500 (好像不对)
T = 1
M = 200
f = lambda s: s
for m in range(1, M+1):
  s = (m - 1) / M
  h = s / M
  sm = m * h
  U_H0 = expm(-1j*T*h * (1 - f(s)) * H0)
  U_H1 = expm(-1j*T*h *      f(s)  * H1)
  qs = U_H1 @ U_H0 @ qs
# 测量末态
print('qstate final:', qs)


# debug
interact(local=globals())
