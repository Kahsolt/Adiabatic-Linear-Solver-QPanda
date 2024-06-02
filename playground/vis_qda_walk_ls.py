#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/01

# 理解绝热演化用于解线性方程组 |x> = Walk|b>
# - 感觉精度还是不行，需要 T=5000 步数迭代才能 fid=0.99

import numpy as np

from utils import *

# make a posdef A as exmaple
MAGIC = Am.T
A = MAGIC @ Am
b = MAGIC @ bv
assert is_posdef(A)
A = A.astype(np.float32)
b = b.astype(np.complex64)
b_norm = np.linalg.norm(b)
b /= b_norm
A /= b_norm
assert is_posdef(A)
κ = condition_number(A)
print_matrix(A, 'A')
print_matrix(b, 'b')
x_gt = np.linalg.solve(A, b)
x_gt /= np.linalg.norm(x_gt)
print('|x>: ', state_vec(x_gt))

# quantum walk hparams
T = 5000        # walk steps
λ = 1           # just assume BE is perfect :)
ver = 'essay'   # ['essay', 'report', 'no_arcsin']

# AQC(p) schedule
f = make_f_s_AQC_P(2.0, κ)

# arxiv:1805.10549, simplified case when A is posdef
A_s = lambda s: (1 - f(s)) * I_(nq) + f(s) * A
P = I_(nq) - b @ b.conj().T
H_s = lambda s: A_s(s) @ P @ A_s(s)
print_matrix(H_s(0), 'H_s(0)')    # |b> 是 H_s(0) 的特征向量
print_matrix(H_s(1), 'H_s(1)')    # 但 |x> 却不是 H_s(1) 的特征向量，相差一个相对相位 (负号)，怎么回事呢？

# system init/target state
psi_0 = b
psi_1 = x_gt
print('|φ(0)>: ', state_vec(psi_0))
print('|φ(1)>: ', state_vec(psi_1))

# simulation process
psi = psi_0
print('init state:', state_vec(psi))

tht_list, phi_list, info_list = [], [], []
for idx, t in enumerate(range(1, 1+T)):
  # cur state: |φ(t+1)> = W_T(t) |φ(t)>
  # NOTE: this operation is iteratively accumulative!!
  H = H_s(t / T)
  psi = W_T(H / λ, ver) @ psi

  # FIXME: force norm here because BE is not unitary, and maybe precision error accumulating :(
  psi /= np.linalg.norm(psi)

  # data point
  tht, phi = amp_to_bloch(psi)
  tht_list.append(tht)
  phi_list.append(phi)
  if idx % 10 == 0: print(f'[step={t}] tht={tht}, phi={phi}')

  # title
  psi_str = state_str(psi)
  fid = get_fidelity(psi, psi_1)
  info_list.append(f'{psi_str} (fid: {fid:.7f})')

print('final state:', state_vec(psi))
print('fidelity:', get_fidelity(psi, psi_1))

# animation
animate_cheap_bloch_plot(phi_list, tht_list, info_list, title='QDA Walk Linear Solver Demo', fp='qda_walk_ls.mp4')
