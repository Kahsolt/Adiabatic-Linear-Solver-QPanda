#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/29

# 理解哈密顿模拟-虚时演化算子 |φ(t)> = exp(-iHt) |φ(0)>, t: 0 -> +inf
# - 若 |φ(0)> 恰好是系统哈密顿量 H 的一个本征态，则演化算子没啥卵用，系统状态一直维持 |φ(0)>
# - 若 |φ(0)> 不是本征态, 那么系统的含时态矢 |φ(t)> 会转圈圈，且周期不定

import numpy as np
from scipy.linalg import expm

from utils import *

# time evolution hparams
T = np.linspace(0, 10, 1000)   # physical time point for evolution

# system hamiltonion (random)
H = rand_hermitian(2)

# system init state
if not 'eigen state':   # an eigen of H
  psi_0 = eigen_state_of_matrix(H)
else:               # random state
  psi_0 = rand_state()

# simulation process
print('init state:', state_vec(psi_0))

tht_list, phi_list, info_list = [], [], []
for idx, t in enumerate(T):
  # cur state: |φ(t)> = exp(-iHt) |φ(0)>
  # NOTE: to avoid error accumulation, this operation is NOT iteratively accumulative!!
  psi_t = expm(-1j*H*t) @ psi_0

  # data point
  tht, phi = amp_to_bloch(psi_t)
  tht_list.append(tht)
  phi_list.append(phi)
  if idx % 10 == 0: print(f'[time={t}] tht={tht}, phi={phi}')

  # title
  psi_str = state_str(psi_t)
  info_list.append(psi_str)

print('final state:', state_vec(psi_t))

# animation
animate_cheap_bloch_plot(phi_list, tht_list, info_list, title='Time Evolution Demo', fp='time_evolution.mp4')
