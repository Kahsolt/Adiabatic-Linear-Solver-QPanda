#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/09 

# 测试一阶近似对保真度的影响

import numpy as np
from scipy.linalg import expm
from utils import rand_hermitian, rand_state, get_fidelity, print_matrix, I_

T = 0.1   # T 越小越近似
fid_list = []
for _ in range(10000):
  nq = 1
  H = rand_hermitian(N=2**nq).real
  #print_matrix(H, 'H')
  ψ = rand_state()
  #print_matrix(ψ, 'ψ')

  TE = expm(-1j*H*T)
  TE_approx = I_(nq) - 1j*H*T

  ψ0 = TE @ ψ
  ψ1 = TE_approx @ ψ
  ψ1 /= np.linalg.norm(ψ1)    # force re-norm

  fid = get_fidelity(ψ1, ψ0)
  print('fid:', fid)
  fid_list.append(fid)

fids = np.asarray(fid_list)
print('max(fid):', fids.max())
print('min(fid):', fids.min())
print('mean(fid):', fids.mean())
print('std(fid):', fids.std())
