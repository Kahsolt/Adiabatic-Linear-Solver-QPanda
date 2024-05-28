#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/28

# 理解绝热演化过程 H(s) = (1 - f(s)) * H0 + f(s) * H1
# - 演化总时长 T 必须足够大，才符合缓慢演化假设
# - 阶段步数 S 必须足够多，每个时间片才能近似为稳态，此刻的含时哈密顿量 H(s) 才能近似为不含时
# - 一旦态矢开始转圈圈就说明寄了，因为系统状态已经偏离当前 H(s) 的本征态了 :(

from functools import partial
from scipy.linalg import expm

from utils import *

# adiabatic evolution hparams
T = 10000   # physical total time period for evolution
S = 300     # logical steps of hamiltonion change

# system hamiltonion (random)
H0 = rand_hermitian(2)
H1 = rand_hermitian(2)
if not 'linear':    # linear behaves very bad :(
  f = lambda s: s
else:               # AQC(p) from arxiv:1805.10549
  κ = 4.2    # FIXME: 先随便xjb估一个kappa...
  def f_(p:float, s:float) -> float:
    t = 1 + s * (κ**(p-1) - 1)
    return κ / (κ - 1) * (1 - t**(1 / (1 - p)))
  f = partial(f_, 2.0)
H_s = lambda s: (1 - f(s)) * H0 + f(s) * H1

# system init/target state
psi_0 = eigen_state_of_matrix(H0, 0)
psi_1 = eigen_state_of_matrix(H1, 0)
print('|φ(0)>: ', state_vec(psi_0))
print('|φ(1)>: ', state_vec(psi_1))

# simulation process
psi = psi_0
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
  fid = get_fidelity(psi, psi_1)
  info_list.append(f'{psi_str} (fid: {fid:.7f})')

print('final state:', state_vec(psi))
print('fidelity:', get_fidelity(psi, psi_1))

# animation
animate_cheap_bloch_plot(phi_list, tht_list, info_list, title='Adiabatic Evolution Demo', fp='adiabatic_evolution.mp4')
