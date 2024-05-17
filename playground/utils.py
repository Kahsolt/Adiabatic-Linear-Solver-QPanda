#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/16

import numpy as np
from numpy import ndarray

''' Const '''

v0 = np.asarray([[1, 0]]).T   # |0>
v1 = np.asarray([[0, 1]]).T   # |1>
h0 = np.asarray([[1,  1]]).T / np.sqrt(2)   # |+>
h1 = np.asarray([[1, -1]]).T / np.sqrt(2)   # |->
σi = np.asarray([   # pauli-i
  [1, 0],
  [0, 1],
])
σx = np.asarray([   # pauli-x
  [0, 1],
  [1, 0],
])
σy = np.asarray([   # pauli-y
  [0, -1j],
  [1j, 0],
])
σz = np.asarray([   # pauli-z
  [1, 0],
  [0, -1],
])
σp = (σx + 1j*σy) / 2   # σ+
σm = (σx - 1j*σy) / 2   # σ-

# equation
Am = np.asarray([
  [2, 1],
  [1, 0],
])
bv = np.asarray([[3, 1]]).T
N = Am.shape[0]
# normalize both side by |b|
b_norm = np.linalg.norm(bv)
A = Am / b_norm
b = bv / b_norm
# target final state |x>, eq. to |+>
x = np.asarray([[1, 1]]).T / np.sqrt(2)


''' Utils '''

is_posdef = lambda A: all([ev > 0 for ev in np.linalg.eigvals(A)])
is_hermitian = lambda H: np.allclose(H, H.conj().T)
is_unitary = lambda U: np.allclose((U.conj().T @ U).real, np.eye(U.shape[0]), atol=1e-6)

def assert_hermitian(H:ndarray):
  assert is_hermitian(H), 'matrix should be hermitian'

def assert_unitary(U:ndarray):
  assert is_unitary(U), 'matrix should be unitary'

def spectral_norm(A:ndarray) -> float:
  '''
  spectral norm (p=2) for matrix 
    non-square: ||A||2 = sqrt(λmax(A*A)) sqrt(A*A的最大特征值) = σmax(A) 最大奇异值
    square: ||A||2 = λmax 最大特征值
  '''
  evs = (np.linalg.eigvalsh if is_hermitian(A) else np.linalg.eigvals)(A)
  return max(np.abs(evs))

def condition_number(A:ndarray) -> float:
  '''
  https://www.phys.uconn.edu/~rozman/Courses/m3511_18s/downloads/condnumber.pdf
  NOTE: condition_number(A) = 1 for unitary A
  '''
  evs = (np.linalg.eigvalsh if is_hermitian(A) else np.linalg.eigvals)(A)
  λmax = max(np.abs(evs))
  λmin = min(np.abs(evs))
  return λmax / λmin

κ = condition_number(A)   # 5.82842712474619
ε = 0.01

def print_matrix(A:ndarray, name:str='A'):
  if len(A.shape) == 2 and min(A.shape) > 1:
    print(f'{name}: (norm={spectral_norm(A):.4f}, κ={condition_number(A):.4f}, shape={A.shape})')
  else:
    print(f'{name}: (norm={np.linalg.norm(A):.4f}, shape={A.shape})')
  print(A.round(4))
