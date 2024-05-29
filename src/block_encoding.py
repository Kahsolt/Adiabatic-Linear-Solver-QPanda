#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/20

from typing import List
import numpy as np
from numpy import ndarray
from pyqpanda import CPUQVM, QVec
from pyqpanda import QCircuit, H, I, X, Y, Z, RY, RZ, CNOT, SWAP, BARRIER
from pyqpanda import QProg, draw_qprog
from pyqpanda import matrix_decompose_hamiltonian, amplitude_encode, Encode
from pyqpanda import matrix_decompose, DecompositionMode

DEBUG = False
QCIR_SAVE = False
QCIR_LEN = 200

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

# predefined matrices for test
A_complex = np.asarray([
  [0.1+0.2j, -0.3],
  [0, -0.4j],
])
A_real = np.asarray([
  [0.1, 0],
  [-0.2, 0.3],
])
# PauliY is not supported by pyQPanda :(
A_LCU = 0.1 * PauliI + 0.2 * PauliX + 0.4 * PauliZ


def show_circuit_matrix_UA(A:ndarray, qvm:CPUQVM, qv:QVec, qcir:QCircuit, lmbd:float):
  n_qubit_ex = len(qv)
  N_ex = 2 ** n_qubit_ex
  U_A = np.zeros([N_ex, N_ex], dtype=np.complex64)
  for i in range(N_ex):
    r = i
    cond = QCircuit()
    for j in range(n_qubit_ex):
      if r & 1: cond << X(qv[j])
      r >>= 1
    qprog = QProg() << cond << BARRIER(qv) << qcir
    qvm.directly_run(qprog)
    qs = qvm.get_qstate()
    U_A[:, i] = np.asarray(qs)

  if np.abs(U_A.imag).max() < 1e-5:
    U_A = U_A.real
  N = A.shape[0]
  U_A = U_A[:N, :N]

  print('A:')
  print(A.round(4))
  print('U_A:')
  print(U_A.round(4))
  print('U_A * lmbd:')
  print((U_A * lmbd).round(4))


def block_encoding_QSVT(A:ndarray) -> ndarray:
  def sqrt_matrix(A:ndarray):
    evs, vecs = np.linalg.eigh(A)
    evs = np.real(evs)
    evs = np.where(evs > 0.0, evs, 0.0)
    evs = evs.astype(vecs.dtype)
    return vecs @ np.diag(np.sqrt(evs)) @ vecs.conj().T

  N = A.shape[0]
  At = A.conj().T
  nq = int(np.log2(A.shape[0]))
  I = np.eye(2**nq)
  U_A = np.zeros_like(np.eye(2**(nq+1)), dtype=np.complex64)
  U_A[:N, :N] = A
  U_A[:N, N:] = sqrt_matrix(I - A @ At)
  U_A[N:, :N] = sqrt_matrix(I - At @ A)
  U_A[N:, N:] = -At

  qvm = CPUQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(nq + 1)
  qcir = matrix_decompose(qv, U_A, DecompositionMode.QSDecomposition)

  print(draw_qprog(qcir, line_length=QCIR_LEN))
  if QCIR_SAVE: draw_qprog(qcir, line_length=QCIR_LEN, output='pic', filename='qcir-QSVT.png')
  if DEBUG: show_circuit_matrix_UA(A, qvm, qv, qcir, lmbd=1)

print('[block_encoding_QSVT]')
block_encoding_QSVT(A_complex)


def block_encoding_LCU(A:ndarray) -> ndarray:
  ops = matrix_decompose_hamiltonian(A)
  if DEBUG: print('LCU:', ops)
  coeffs = []
  terms = []
  for what, coeff in ops.data():
    coeffs.append(coeff.real)
    terms.append(what[-1][0] if what[-1] else 'I')
  if DEBUG:
    print('coeffs:', coeffs)
    print('terms:', terms)

  n_qubit = int(np.ceil(np.log2(A.shape[0])))
  n_unitary = len(coeffs)
  n_ancilla = int(np.ceil(np.log2(n_unitary)))
  n_qubit_ex = n_qubit + n_ancilla
  if DEBUG:
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
  probs = np.abs(coeffs) / lmbd
  amplitude = np.sqrt(probs)
  if DEBUG:
    print('lmbd:', lmbd)
    print('probs:', probs, 'sum=', probs.sum())
    print('amplitude:', amplitude)

  if not 'use Encode':
    encoder = Encode()
    encoder.amplitude_encode(qv_a, amplitude)
    PREP = encoder.get_circuit()
  else:
    PREP = amplitude_encode(qv_a, amplitude)
  PREP_dagger = PREP.dagger()

  qprog = QProg() << PREP
  probs = qvm.prob_run_list(qprog, qv_a)
  if DEBUG: print('probs (PREP):', probs)

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
  print(draw_qprog(qcir, line_length=QCIR_LEN))
  if QCIR_SAVE: draw_qprog(qcir, line_length=QCIR_LEN, output='pic', filename='qcir-LCU.png')
  if DEBUG: show_circuit_matrix_UA(A, qvm, qv, qcir, lmbd)

print('[block_encoding_LCU]')
block_encoding_LCU(A_LCU)


def block_encoding_ARCSIN(A:ndarray, tol:float=1e-5) -> ndarray:
  ''' arxiv:2402.17529, the basis QUERY-ORACLE framework of FABLE '''
  N_ROW, N_COL = A.shape
  nq_row = int(np.ceil(np.log2(N_ROW)))
  nq_col = int(np.ceil(np.log2(N_COL)))
  n_qubit = max(nq_row, nq_col)
  n_qubit_ex = 1 + n_qubit * 2
  if DEBUG:
    print('n_qubit:', n_qubit)
    print('n_qubit_ex:', n_qubit_ex)
  lmbd = 2**n_qubit

  qvm = CPUQVM()
  qvm.init_qvm()
  # |c0,...,cn;r0,...,rm;a>
  qv = qvm.qAlloc_many(n_qubit_ex)
  # Fig. 3 from arxiv:2402.17529, mind the fucking order!!
  qv_col = qv[:n_qubit]
  qv_row = qv[n_qubit:-1]
  q_anc = qv[-1]
  qv_mctrl = qv[:-1]
  if DEBUG:
    print('len(qv_row):', len(qv_row))
    print('len(qv_col):', len(qv_col))
    print('len(qv_mctrl):', len(qv_mctrl))

  # Eq. 15 & 16 from arxiv:2205.00081, the basic "matrix query oracle" in FABLE essay
  # a_{i,j} = |a_{i,j}| exp(i * \alpha_{i,j})
  # \theta_{i,j} = arccos(|a_{i,j}|)
  # \phi_{i,j} = -\alpha_{i,j}
  # Eq. 6 from arxiv:2402.17529, here we use ARCSIN method for the real parts
  thetas = np.arcsin(np.abs(A))
  phis = np.angle(A)
  if DEBUG:
    print('thetas:', thetas)
    print('phis:', phis)

  qcir = QCircuit()
  for q in qv_row: qcir << H(q)
  qcir << BARRIER(qv)
  for i in range(N_ROW):
    for j in range(N_COL):
      t = 2 * thetas[i, j]
      p = 2 * phis  [i, j]
      if abs(t) < tol and abs(p) < tol: continue

      # cond
      cond = QCircuit()
      loc = i
      for q_r in qv_row:
        if loc & 1 == 0: cond << X(q_r)
        loc >>= 1
      loc = j
      for q_c in qv_col:
        if loc & 1 == 0: cond << X(q_c)
        loc >>= 1
      # mctrl-rot
      mcrot_circ = QCircuit()
      if abs(t) > tol: mcrot_circ << RY(q_anc, t).control(qv_mctrl)
      if abs(p) > tol: mcrot_circ << RZ(q_anc, p).control(qv_mctrl)
      # cond (uncompute)
      qcir << cond << mcrot_circ << cond << BARRIER(qv)
  for q_r, q_c in zip(qv_row, qv_col):
    qcir << SWAP(q_r, q_c)
  qcir << X(q_anc)    # ARCSIN flip ancilla
  qcir << BARRIER(qv)
  for q in qv_row: qcir << H(q)
  print(draw_qprog(qcir, line_length=QCIR_LEN))
  if QCIR_SAVE: draw_qprog(qcir, line_length=QCIR_LEN, output='pic', filename='qcir-ARCSIN.png')
  if DEBUG: show_circuit_matrix_UA(A, qvm, qv, qcir, lmbd)

print('[block_encoding_ARCSIN]')
block_encoding_ARCSIN(A_complex)


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
  if DEBUG: print('M_trans:', M_trans)
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

def block_encoding_FABLE(A:ndarray, tol:float=1e-8) -> ndarray:
  n_qubit = int(np.ceil(np.log2(A.shape[0])))
  n_qubit_ex = 1 + n_qubit * 2
  if DEBUG:
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
  if DEBUG:
    print('wires:', wires)
    print('ancilla:', ancilla)
    print('wires_i:', wires_i)
    print('wires_j:', wires_j)
  
  code = block_encoding_FABLE_gray_code(2 * n_qubit)
  if DEBUG: print('code:', code)
  n_sel = len(code)
  ctrl_wires = [
    int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % n_sel], 2)))
    for i in range(n_sel)
  ]
  wire_map = dict(enumerate(wires_j + wires_i))
  if DEBUG:
    print('ctrl_wires:', ctrl_wires)
    print('wire_map:', wire_map)

  alphas = np.arccos(A).flatten()
  thetas = block_encoding_FABLE_compute_theta(alphas)
  if DEBUG:
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
  print(draw_qprog(qcir, line_length=QCIR_LEN))
  if QCIR_SAVE: draw_qprog(qcir, line_length=QCIR_LEN, output='pic', filename='qcir-FABLE.png')
  if DEBUG: show_circuit_matrix_UA(A, qvm, qv, qcir, lmbd)

print('[block_encoding_FABLE]')
block_encoding_FABLE(A_real)
