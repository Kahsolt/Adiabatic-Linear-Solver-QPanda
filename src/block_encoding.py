#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/20

from typing import List
import numpy as np
from numpy import ndarray
from pyqpanda import CPUQVM, QVec
from pyqpanda import QCircuit, H, I, X, Y, Z, RY, CNOT, SWAP, BARRIER
from pyqpanda import QProg, draw_qprog
from pyqpanda import matrix_decompose_hamiltonian, amplitude_encode, Encode

PauliI = np.asarray([
  [1, 0],
  [0, 1],
])
PauliX = np.asarray([
  [0, 1],
  [1, 0],
])
PauliY = np.asarray([
  [0, -1j],
  [1j, 0],
])
PauliZ = np.asarray([
  [1, 0],
  [0, -1],
])


def block_encoding_LCU() -> ndarray:
  # PauliY is not supported by pyQPanda
  A = 0.1 * PauliI + 0.2 * PauliX + 0.4 * PauliZ
  #A = 0.774304 * PauliX

  ops = matrix_decompose_hamiltonian(A)
  print('LCU:', ops)
  coeffs = []
  terms = []
  for what, coeff in ops.data():
    coeffs.append(coeff.real)
    terms.append(what[-1][0] if what[-1] else 'I')
  print('coeffs:', coeffs)
  print('terms:', terms)

  n_qubit = int(np.ceil(np.log2(A.shape[0])))
  n_unitary = len(coeffs)
  n_ancilla = int(np.ceil(np.log2(n_unitary)))
  n_qubit_ex = n_qubit + n_ancilla
  print('n_unitary:', n_unitary)
  print('n_ancilla:', n_ancilla)
  print('n_qubit_ex:', n_qubit_ex)

  qvm = CPUQVM()
  qvm.init_qvm()
  # |a,w>
  qv_w = qvm.qAlloc_many(n_qubit)
  qv_a = qvm.qAlloc_many(n_ancilla)
  qv = qv_w + qv_a

  lmbd = np.abs(coeffs).sum()
  print('lmbd:', lmbd)
  probs = np.abs(coeffs) / lmbd
  print('probs:', probs, 'sum=', probs.sum())
  amplitude = np.sqrt(probs)
  print('amplitude:', amplitude)

  if n_unitary == 1:
    U_A = A / lmbd
    print('A:')
    print(A.round(4))
    print('U_A:')
    print(U_A.round(4))
    print('U_A * lmbd:')
    print((U_A * lmbd).round(4))
    return U_A

  if not 'use Encode':
    encoder = Encode()
    encoder.amplitude_encode(qv_a, amplitude)
    PREP = encoder.get_circuit()
  else:
    PREP = amplitude_encode(qv_a, amplitude)
  PREP_dagger = PREP.dagger()

  qprog = QProg() << PREP
  probs = qvm.prob_run_list(qprog, qv_a)
  print('probs (PREP):', probs)

  SEL = QCircuit()
  for i in range(len(terms)):
    r = i
    cond = QCircuit()
    for j in range(n_ancilla):
      if r & 1 == 0: cond << X(qv_a[j])
      r >>= 1
    cond_dagger = cond.dagger()
    term = terms[i]
    cu = QCircuit()
    j = 0
    sym = term[j]
    if sym == 'I': cu << I(qv_w[j]).control(qv_a)
    if sym == 'X': cu << X(qv_w[j]).control(qv_a)
    if sym == 'Z': cu << Z(qv_w[j]).control(qv_a)
    SEL << cond << cu << cond_dagger << BARRIER(qv)

  qcir = PREP << BARRIER(qv) << SEL << PREP_dagger
  print(qcir)
  #draw_qprog(qcir, output='pic', filename='qcir.png')

  N_ex = 2 ** n_qubit_ex
  U_A = np.zeros([N_ex, N_ex])
  for i in range(N_ex):
    r = i
    cond = QCircuit()
    for j in range(n_qubit_ex):
      if r & 1: cond << X(qv[j])
      r >>= 1
    qprog = QProg() << cond << qcir
    qvm.directly_run(qprog)
    qs = qvm.get_qstate()
    U_A[:, i] = np.asarray(qs).real

  print('A:')
  print(A.round(4))
  print('U_A:')
  print(U_A.round(4))
  print('U_A * lmbd:')
  print((U_A * lmbd).round(4))

block_encoding_LCU()
print()


def block_encoding_FABLE_compute_theta(alpha:ndarray) -> ndarray:
    def _matrix_M_entry(row, col):
      # (col >> 1) ^ col is the Gray code of col
      b_and_g = row & ((col >> 1) ^ col)
      sum_of_ones = 0
      while b_and_g > 0:
        if b_and_g & 0b1:
          sum_of_ones += 1
        b_and_g = b_and_g >> 1
      return (-1) ** sum_of_ones

    ln = alpha.shape[-1]
    k = np.log2(ln)
    M_trans = np.zeros(shape=(ln, ln))
    for i in range(len(M_trans)):
      for j in range(len(M_trans[0])):
        M_trans[i, j] = _matrix_M_entry(j, i)
    print('M_trans:', M_trans)
    theta = np.transpose(np.dot(M_trans, np.transpose(alpha)))
    return theta / 2**k

def block_encoding_FABLE_gray_code(rank:int) -> List[str]:
  def gray_code_recurse(g, rank):
    k = len(g)
    if rank <= 0: return

    for i in range(k - 1, -1, -1):
      char = "1" + g[i]
      g.append(char)
    for i in range(k - 1, -1, -1):
      g[i] = "0" + g[i]
    gray_code_recurse(g, rank - 1)

  g = ["0", "1"]
  gray_code_recurse(g, rank - 1)
  return g

def block_encoding_FABLE(tol:float=1e-8) -> ndarray:
  A = np.asarray([
    [0.1, 0],
    [-0.2, 0.3],
  ])

  n_qubit = int(np.ceil(np.log2(A.shape[0])))
  n_qubit_ex = 1 + n_qubit * 2
  print('n_qubit:', n_qubit)
  print('n_qubit_ex:', n_qubit_ex)

  qvm = CPUQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(n_qubit_ex)
  # |a,i,j(working)>
  wires = list(reversed(range(n_qubit_ex)))
  ancilla = wires[0]
  wires_i = wires[1 : 1 + len(wires) // 2][::-1]
  wires_j = wires[1 + len(wires) // 2 : len(wires)][::-1]
  print('wires:', wires)
  print('ancilla:', ancilla)
  print('wires_i:', wires_i)
  print('wires_j:', wires_j)
  
  code = block_encoding_FABLE_gray_code(2 * n_qubit)
  print('code:', code)
  n_sel = len(code)
  ctrl_wires = [
    int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % n_sel], 2)))
    for i in range(n_sel)
  ]
  print('ctrl_wires:', ctrl_wires)
  wire_map = dict(enumerate(wires_j + wires_i))
  print('wire_map:', wire_map)

  alphas = np.arccos(A).flatten()
  thetas = block_encoding_FABLE_compute_theta(alphas)
  print('alphas:', alphas)
  print('thetas:', thetas)
  lmbd = 2**n_qubit

  qcir = QCircuit()
  for i in wires_i: qcir << H(qv[i])
  qcir << BARRIER(qv)
  nots = set()
  for theta, ctrl_idx in zip(thetas, ctrl_wires):
    if abs(2 * theta) > tol:
      for ctrl_wire in nots:
        qcir << CNOT(qv[ctrl_wire], qv[ancilla])
      qcir << RY(qv[ancilla], 2 * theta)
      nots.clear()
    if wire_map[ctrl_idx] in nots:
      nots.remove(wire_map[ctrl_idx])
    else:
      nots.add(wire_map[ctrl_idx])
  for ctrl_wire in nots:
    qcir << CNOT(qv[ctrl_wire], qv[ancilla])
  qcir << BARRIER(qv)
  for i, j in zip(wires_i, wires_j):
    qcir << SWAP(qv[i], qv[j])
  qcir << BARRIER(qv)
  for i in wires_i: qcir << H(qv[i])
  print(qcir)

  N_ex = 2 ** n_qubit_ex
  U_A = np.zeros([N_ex, N_ex])
  for i in range(N_ex):
    r = i
    cond = QCircuit()
    for j in range(n_qubit_ex):
      if r & 1: cond << X(qv[j])
      r >>= 1
    qprog = QProg() << cond << qcir
    qvm.directly_run(qprog)
    qs = qvm.get_qstate()
    U_A[:, i] = np.asarray(qs).real

  print('A:')
  print(A.round(4))
  print('U_A:')
  print(U_A.round(4))
  print('U_A * lmbd:')
  print((U_A * lmbd).round(4))

block_encoding_FABLE()
print()
