#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/06 

# reproducing the QDA paper (arXiv:2111.08152 & arXiv:2312.07690) in pennylane
# Fig. 4, qreg: |ah,a1,a2,a,ψ>

import numpy as np
from pennylane import *
from pennylane.measurements import StateMP

from utils import A, b, κ, make_f_s_AQC_P, print_matrix
from utils import Am, bv, is_posdef, condition_number

if 'magic transform':
  MAGIC = Am.T
  A = MAGIC @ Am
  b = MAGIC @ bv
  assert is_posdef(A)
  A = A.astype(np.float32)
  b = b.astype(np.float32)
  b_norm = np.linalg.norm(b)
  b /= b_norm
  A /= b_norm
  assert is_posdef(A)
  κ = condition_number(A)

# data
print_matrix(A, 'A')
print_matrix(b, 'b')
# encode |b> angle
theta_b = 2 * np.arccos(b[0].item())
print('theta_b:', theta_b)
# sched
f_s = make_f_s_AQC_P(2, κ)

# pennylane qubit order: |wire_0, wire_1, ..., wire_q>
n_qubits = 5
dev = device('default.qubit', wires=['ah', 'a1', 'a2', 'a', 'ψ'])

@prod
def R_s(s:float):
  '''
  R(s) = [
    [1-f(s), f(s)]
    [f(s), -(1-f(s))]
  ] / sqrt((1-f)**2 + f**2)
  X * RY = [
    [sin(θ/2), cos(θ/2)],
    [cos(θ/2), -sin(θ/2)],
  ]
  '''
  global f_s
  # f(s) / norm == cos(θ/2)
  f = f_s(s)
  theta_f_s = 2 * np.arccos(f / np.sqrt((1 - f)**2 + f**2))
  RY(theta_f_s, wires='a1')
  X(wires='a1')

@prod
def BE_Hs(s:float):
  global A, b
  # H
  Hadamard(wires='a2')
  # CU_Qb
  RY(theta_b, wires='ψ')
  ctrl(PauliZ(wires='ψ'), control=['ah', 'a2'])
  adjoint(RY(theta_b, wires='ψ'))
  # CR(s)
  ctrl(R_s(s), control='ah', control_values=False)
  ctrl(Hadamard(wires='a1'), control='ah')
  # U_A(f)
  ctrl(BlockEncode(A, wires=['a', 'ψ']), control=['ah', 'a1'])
  PauliX(wires='ah')
  ctrl(adjoint(BlockEncode(A, wires=['a', 'ψ'])), control=['ah', 'a1'])
  # CR(s)
  ctrl(Hadamard(wires='a1'), control='ah')
  ctrl(R_s(s), control='ah', control_values=False)
  # CU_Qb
  RY(theta_b, wires='ψ')
  ctrl(PauliZ(wires='ψ'), control=['ah', 'a2'])
  adjoint(RY(theta_b, wires='ψ'))
  # H
  Hadamard(wires='a2')

@qnode(dev, interface=None, grad_on_execution=True, diff_method=None)
def QDA(S:int=200) -> StateMP:
  for s in range(S):
    # W_Ts
    Reflection(BE_Hs(s / S), reflection_wires=['a1', 'a2', 'a'])
  return state()


'''
|x>: [0.42467473+1.14852967e-13j 0.57499991+7.66275448e-14j]
|x> renorm: [0.59409681+1.60673043e-13j 0.80439355+1.07197761e-13j]
fidelity: 0.9888820162025821
'''
qs = QDA(5000)
print('qs:', qs.round(4))
x = np.asarray([qs[0], qs[8]])
print('|x>:', x)
x = x / np.linalg.norm(x)
print('|x> renorm:', x)
x_ideal = np.asarray([1, 1])
x_ideal = x_ideal / np.linalg.norm(x_ideal)
print('fidelity:', np.abs(np.dot(x.T, x_ideal)))
