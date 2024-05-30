#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/30 

# 理解特征过滤 |H_λ> P_λ|ψ>
#  - P_λ 是矩阵 H 的一个投影算子，能从任意向量中投影出对应于矩阵 H 第 λ 个零空间向量 |H_λ> 的成分

from utils import *

# 系统末态哈密顿量 H1
Qb = I_(nq) - b @ b.conj().T
H1 = np.kron(σp, A @ Qb) + np.kron(σm, Qb @ A)
print_matrix(H1, 'H1')
# H1 的零空间
H1_nullspace = [    # H1 @ null(H1) = 0
  np.kron(v0, x),   # \tilde{x}; if A|x> ~ |b>, then Qb*A|x> = Qb*|b> = 0
  np.kron(v1, b),   # \bar{b}
]

# AQC 的期望末态
v = H1_nullspace[0]
print_matrix(v, 'v')
assert np.allclose(H1 @ v, np.zeros_like(v))

# 假设 AQC 得到的是一个近似解
v1 = v + np.random.uniform(size=v.shape, low=-1, high=1) * 0.2
v1 /= np.linalg.norm(v1)
print_matrix(v1, 'v1')
fid_v1 = get_fidelity(v1, v)
print('fid(v1):', fid_v1)

# 制作投影矩阵 P0，神奇的是这玩意儿不保模长
Pλ = R_2l(H1)
print_matrix(Pλ, 'Pλ')

# 过滤！
v2 = v1
for i in range(10):    # 这是可以迭代的
  v2 = Pλ @ v2
  v2 /= np.linalg.norm(v2)
  fid_v2 = get_fidelity(v2, v)
  print('fid(v2):', fid_v2)

print_matrix(v2, 'v2')
fid_v2 = get_fidelity(v2, v)
assert fid_v2 > fid_v1
print('improve:', fid_v2 - fid_v1)
