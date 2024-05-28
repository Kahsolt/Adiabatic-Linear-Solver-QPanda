#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/16

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy import ndarray

BASE_PATH = Path(__file__).parent.parent
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)

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
H = np.asarray([
  [1,  1],
  [1, -1],
]) / np.sqrt(2)
RY = lambda θ: np.asarray([   # e^(-i*Y*θ/2)
  [np.cos(θ/2), -np.sin(θ/2)],
  [np.sin(θ/2),  np.cos(θ/2)],
])
I_ = lambda nq: np.eye(2**nq)

# equation
Am = np.asarray([
  [2, 1],
  [1, 0],
])
bv = np.asarray([[3, 1]]).T
if not 'magic transform':
  # left multiply Am.T can turn Am to be posdef :)
  # but heavily enlarge cond_num 5.828 -> 33.970 :(
  MAGIC = Am.T
  Am = MAGIC @ Am
  bv = MAGIC @ bv
N = Am.shape[0]
nq = int(np.log2(N))
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
    print(f'{name}: (norm={spectral_norm(A):.4g}, κ={condition_number(A):.4g}, shape={A.shape})')
  else:
    print(f'{name}: (norm={np.linalg.norm(A):.4g}, shape={A.shape})')
  print(A.round(4))


''' Visual Utils '''

Stat = ndarray
Ham = ndarray

def rand_state() -> Stat:
  psi = np.empty([2, 1], dtype=np.complex64)
  psi.real = np.random.uniform(size=psi.shape)
  psi.imag = np.random.uniform(size=psi.shape)
  psi /= np.linalg.norm(psi)
  return psi

def rand_hermitian(N:int) -> Ham:
  A = np.empty([N, N], dtype=np.complex64)
  A.real = np.random.uniform(size=A.shape)
  A.imag = np.random.uniform(size=A.shape)
  H = A.conj().T @ A    # make it hermitian!
  return H

def eigen_state_of_matrix(A:ndarray, id:int=-1) -> ndarray:
  eigvecs = np.linalg.eig(A)[1]
  if id < 0:
    n_eigvecs = eigvecs.shape[-1]
    id = random.randrange(n_eigvecs)
  return np.expand_dims(eigvecs[:, id], -1)   # [N, 1]

def state_str(psi:Stat) -> str:
  psi = drop_gphase(psi)
  a = psi[0].item().real
  c = psi[1].item().real
  d = psi[1].item().imag
  sign1 = '+' if c >= 0 else '-'
  if sign1 == '-': c, d = -c, -d
  sign2 = '+' if d >= 0 else '-'
  if sign2 == '-': d = -d
  return f'{a:.3f} |0> {sign1} ({c:.3f} {sign2} {d:.3f}i) |1>'

def state_vec(psi:Stat) -> Stat:
  return psi.T[0]

def drop_gphase(psi:Stat) -> Stat:
  return psi * (psi[0].conj() / np.abs(psi[0]))

def amp_to_bloch(psi:Stat) -> Tuple[float, float]:
  psi = drop_gphase(psi).T[0]
  tht = np.arccos(psi[0].real)
  phi = np.angle(psi[1])
  return tht.item(), phi.item()

def get_fidelity(psi:Stat, phi:Stat) -> float:
  return np.abs(np.dot(psi.T, phi)).item()


def animate_cheap_bloch_plot(xlist:List[float], ylist:List[float], tlist:List[str], auto_lim:bool=False, title:str='Demo', fp:Path=None):
  import matplotlib.pyplot as plt
  from matplotlib.axes import Axes
  from matplotlib.animation import FuncAnimation 

  # https://stackoverflow.com/questions/42722691/python-matplotlib-update-scatter-plot-from-a-function
  def update(s:int):
    nonlocal ax, sc
    sc.set_offsets(np.c_[xlist[:s], ylist[:s]])
    ax.set_title(tlist[s])

  print('>> Exporting video... this takes long, please wait...')
  fig, ax = plt.subplots()
  ax: Axes
  sc = ax.scatter([], [], s=3)
  plt.gca().invert_yaxis()
  plt.xlabel('phi (phase)')
  plt.ylabel('theta (polarity)')
  if auto_lim:
    plt.xlim(min(xlist)-0.1, max(xlist)+0.1)
    plt.ylim(min(ylist)-0.1, max(ylist)+0.1)
  else:
    plt.xlim(-np.pi-0.1, np.pi  +0.1)
    plt.ylim(     0-0.1, np.pi/2+0.1)
  plt.suptitle(title)
  anim = FuncAnimation(fig, update, frames=len(xlist), interval=10, repeat=False, cache_frame_data=bool(fp))
  if fp:
    anim.save(fp, fps=60, dpi=400)
    print(f'>> Saved to {fp}')
  else:
    plt.show()
