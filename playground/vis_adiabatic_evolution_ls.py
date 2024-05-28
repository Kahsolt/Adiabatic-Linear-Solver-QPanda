#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/28

# 理解绝热演化用于解线性方程组 |x> = U_{AD}|b>

from functools import partial
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation 

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

# adiabatic evolution hparams
S = 1000
T_ref = κ**2 / ε    # 这里的 T 相当的大 ~11.5w，可能是哈密顿量 H_s 编码方式导致的
T = int(T_ref)
print(f'T: {T} (ref: {T_ref})')

# arxiv:1805.10549, AQC(p) schedule
def f_(p:float, s:float) -> float:
  t = 1 + s * (κ**(p-1) - 1)
  return κ / (κ - 1) * (1 - t**(1 / (1 - p)))
f = partial(f_, 2.0)

# arxiv:1805.10549, simplified case when A is posdef
A_s = lambda s: (1 - f(s)) * I_(nq) + f(s) * A
P = I_(nq) - b @ b.conj().T
H_s = lambda s: A_s(s) @ P @ A_s(s)
print_matrix(H_s(0), 'H_s(0)')    # |b> 是 H_s(0) 的特征向量
print_matrix(H_s(1), 'H_s(1)')    # 但 |x> 却不是 H_s(1) 的特征向量，相差一个相对相位 (负号)，怎么回事呢？

# simulation process
psi = b
print('init state:', state_vec(psi))

tht_list, phi_list, info_list = [], [], []
for idx, s in enumerate(range(S)):
  # cur state: |φ(t+Δt)> = exp(-iH(s)Δt) |φ(t)>
  # NOTE: this operation is iteratively accumulative!!
  Δt = T / S
  H = H_s(s / S)
  psi = expm(-1j*H*Δt) @ psi

  # data point
  tht, phi = amp_to_bloch(psi)
  tht_list.append(tht)
  phi_list.append(phi)
  if idx % 10 == 0: print(f'[step={s}] tht={tht}, phi={phi}')

  # title
  psi_str = state_str(psi)
  fid = get_fidelity(psi, x_gt)
  info_list.append(f'{psi_str} (fid: {fid:.7f})')

print('final state:', state_vec(psi))
print('fidelity:', get_fidelity(psi, x_gt))

# animation
animate_cheap_bloch_plot(phi_list, tht_list, info_list, auto_lim=True, title='Adiabatic Evolution Linear Solver Demo', fp='adiabatic_evolution_ls.mp4')
