#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/15

# 第一问: 块编码问题，对给定哈密顿量 H 找到酉矩阵 U_H 其满足性质: U_H|G,ψ> = |G>H|ψ> + sqrt(1-||H|ψ>||^2) |G_ψ^⊥>
# arXiv:1511.02306  Chap 4
# arXiv:1610.06546  Chap 3.2 & Appendix. A
# arXiv:2111.08152  Appendix. E
# arXiv:2203.10236
# arXiv:2205.00081

from typing import List, Tuple

import pennylane as qml
from pennylane.measurements import StateMP

from utils import *

# H is hermitian, and max. |λ|<1
H = A
print_matrix(H, 'H')
assert_hermitian(H)
assert np.linalg.eigvalsh(H)[0] < 1, '||H||2 should < 1'
print()


''' Method 1: directly construct (cannot generally prepare by a circuit below O(poly(N))) '''

def sqrt_matrix(A:ndarray):
  # ~pennylane.math.quantum.sqrt_matrix
  # sqrt(A) = sqrt(PΛP*) = P @ sqrt(Λ) @ P*
  evs, vecs = np.linalg.eigh(A)
  evs = np.real(evs)
  evs = np.where(evs > 0.0, evs, 0.0)
  evs = evs.astype(vecs.dtype)
  return vecs @ np.diag(np.sqrt(evs)) @ vecs.conj().T

def qsvt_construct_method(A:ndarray) -> ndarray:
  # https://pennylane.ai/qml/demos/tutorial_intro_qsvt/
  # arXiv:2203.10236 Eq. 3.3
  At = A.conj().T
  nq = int(np.log2(A.shape[0]))
  I = I_(nq)
  U_A = np.zeros_like(I_(nq+1))
  U_A[:N, :N] = A
  U_A[:N, N:] = sqrt_matrix(I - A @ At)
  U_A[N:, :N] = sqrt_matrix(I - At @ A)
  U_A[N:, N:] = -At
  return U_A

print('=' * 72)
print('[Method 1: directly construct]')
U_H = qsvt_construct_method(H)
print_matrix(U_H, 'U_H')
assert_unitary(U_H)
print()


''' Method 2: LCU (linear combinations of unitaries) '''
# https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding

def lcu_decompose_pyqpanda(A:ndarray) -> List[Tuple[float, str]]:
  from pyqpanda import CPUQVM
  from pyqpanda import matrix_decompose_paulis

  # decompose hermitian to a pauli-sum
  qvm = CPUQVM() ; qvm.init_qvm()
  paulis = matrix_decompose_paulis(qvm, A)
  print('len(paulis):', len(paulis))
  for coeff, qcir in paulis:
    print('coeff:', coeff)
    print(qcir)
  composed_paulis = 0.31622776601683794 * (σx + σi + σz)
  print('composed_paulis:')
  print(composed_paulis)
  assert np.allclose(A, composed_paulis), 'cannot compose back'

  return [
    (0.31622776601683794, 'X'),
    (0.31622776601683794, 'I'),
    (0.31622776601683794, 'Z'),
  ]

dev = qml.device('default.qubit', wires=3)
@qml.qnode(dev)
def lcu_method(A:ndarray) -> StateMP:
  # 构造 U = PREP* @ SEL @ PREP，即有 <0|U|0>|ψ> = A/λ |ψ>；除去规范因子 λ 的意义下，U 即为 A 的块编码
  #   - LCU 将 A 分解为 k 项，系数和 λ = Σk |α_k|
  #   - |0> 是 ceil(log2(k)) 个辅助 qubit
  #   - PREP 在辅助比特组上，按 LCU 系数制备一个叠加态，如本例中
  #     - PREP|0> = 0.3162*|00> + 0.3162*|01> + 0.3162*|10>, 注意系数要除λ以归一化
  #     - 用 AmplitudeEncoding 实现，或者 Möttönen [https://arxiv.org/abs/quant-ph/0407010]
  #   - SEL 是 k 个多比特受控 U 门，受到辅助比特组控制，条件即序号 k 的二进制编码，U 门即分解出来的pauli门 σk，如本例中
  #     - (I @ I @ I^n) @ X.control([0, 1]) @ (I @ I @ I^n)   # |00>
  #     - (I @ X @ I^n) @ I.control([0, 1]) @ (I @ X @ I^n)   # |01>, I-gate can be ignored
  #     - (X @ I @ I^n) @ Z.control([0, 1]) @ (X @ I @ I^n)   # |10>

  global λ
  # LCU decompose
  coeffs, ops = qml.pauli_decompose(A).terms()
  print(f"coeffs: {coeffs}")
  print(f"ops: {ops}")
  # normalized square roots of coefficients
  λ = np.linalg.norm(np.sqrt(coeffs))
  alphas = np.sqrt(coeffs) / λ
  alphas = np.pad(alphas, [0, 2**N-len(alphas)])
  # relabeling wires: 0 → 2 (first two bits are ancilla; last bit is working)
  unitaries = [qml.map_wires(op, {0: 2}) for op in ops]
  qml.StatePrep(alphas, wires=[0, 1])     # prepare state with `alphas` to ancilla qubits |00>
  qml.Select(unitaries, control=[0, 1])   # apply multi-controlled unitaries to working qubits |ψ>
  qml.adjoint(qml.StatePrep(alphas, wires=[0, 1]))    # DON'T forget the dagger :D
  return qml.state()

print('=' * 72)
print('[Method 1: LCU method]')
U_H = qml.matrix(lcu_method)(H).real
U_H_scaled = U_H * λ**2
print_matrix(U_H_scaled, 'U_H')
assert_unitary(U_H)
print()


''' Method 3: d-sparse Hamiltonion (matrix access oracles) '''
# https://pennylane.ai/qml/demos/tutorial_block_encoding

def oracle_access_framework(A:ndarray):
  '''
  需要准备两个 Oracle 线路: U_A 和 U_B，一个辅助比特 |0>
    - U_A: 取出矩阵A中的元素，放在辅助比特的振幅上
      - U_A|0>|i>|j> = (A_ij|0> + sqrt(1 - A_ij^2)|1>)|i>|j>
      - 矩阵元素 A_ij 越大，辅助比特投影到|0>的概率越大，所以这种方法适合读取稀疏矩阵
      - 朴素的 U_A 构造方法就是 qml.Select，但需要 O(N^4) 个门
    - U_B: 确保遍历所有的下标组合<i,j>，实际上就是个(宏的) SWAP 门，因为 |i>/|j> 可能是多个 qubit
      - U_B|i>|j> = |j>|i>
  '''

dev = qml.device('default.qubit', wires=3)
@qml.qnode(dev)
def fable_method(A:ndarray) -> StateMP:
  qml.FABLE(A, wires=range(3), tol=0)
  return qml.state()

print('=' * 72)
print('[Method 1: FABLE method]')
s = int(np.ceil(np.log2(N)))
U_H = qml.matrix(fable_method)(H).real
U_H_scaled = U_H * (2**s)
print_matrix(U_H_scaled, 'U_H')
assert_unitary(U_H)
print()
