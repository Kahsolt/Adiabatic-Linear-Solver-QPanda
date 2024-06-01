#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/01

# 理解 QDA[arXiv:2111.08152] 论文中的 QWalk 算子
# - 按照论文推算: W_T = -I + H/λ ≈ -exp(H), 全局相位无关紧要(?)
# - 按照报告视频: W_T = expm(i*arcsin(H/λ))
# - λ 为 BE 引入的缩放因子
# - 好像刚开始几步根本不稳定啊草，平均起来成功的例子不是很多..

import numpy as np

from utils import *

# quantum walk hparams
T = 500         # walk steps
λ = 1           # just assume BE is perfect :)
ver = 'report' # ['essay', 'report', 'no_arcsin']

# system hamiltonion (random)
H0 = rand_hermitian(2)
H1 = rand_hermitian(2)
κ = 4.2     # FIXME: 先随便xjb估一个kappa...
f = make_f_s_AQC_P(2.0, κ)
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
animate_cheap_bloch_plot(phi_list, tht_list, info_list, title='QDA Walk Demo', fp='qda_walk.mp4')
