#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/16

import numpy as np
from numpy import ndarray

''' Const '''

v0 = np.asarray([[1, 0]]).T  # |0>
v1 = np.asarray([[0, 1]]).T  # |1>
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


''' Utils '''

is_hermitian = lambda H: np.allclose(H, H.conj().T)
is_unitary = lambda U: np.allclose((U.conj().T @ U).real, np.eye(U.shape[0]), atol=1e-6)

def assert_hermitian(H:ndarray):
  assert is_hermitian(H), 'matrix should be hermitian'

def assert_unitary(U:ndarray):
  assert is_unitary(U), 'matrix should be unitary'


def print_matrix(A:ndarray, name:str='A'):
  print(f'{name}: (norm={np.linalg.norm(A):.5f}, shape={A.shape})')
  print(A.round(4))
