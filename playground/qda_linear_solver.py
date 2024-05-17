#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/15

# 第二问: 线性方程组问题 A * x = b 转 绝热演化问题 H(s) = (1 - f(s)) * H0 + f(s) * H1
# arXiv:1805.10549: RM-based; 非常好论文，使我光速复现！！
# arXiv:1909.05500: AQC(p), AQC(exp), QAOA; 复现了AQC方法部分，论文公式有小错误
# arXiv:2111.08152 Chap. IV: QDA, Walk operator, Block Encoding H(s)

from functools import partial
from typing import Tuple, Callable

from scipy.linalg import expm
from utils import *


def classical_solution():
  print_matrix(Am, 'Am')
  print_matrix(bv, 'bv')
  x = np.linalg.solve(Am, bv)
  print('x:', x.T[0])

  print('-' * 42)

  print_matrix(A, 'A')
  print_matrix(b, 'b')
  x = np.linalg.solve(A, b)
  print('x:', x.T[0])

print('=' * 42)
print('[classical_solution]')
classical_solution()
print()


'''
[Randomization Method]
1. 准备两个哈密顿量 H0 和 H1
2. 将线性系统 Ax=b 的解 x 编码为 H1 的一个本征向量 |x>
3. 制备系统初态 |φ> 为 H0 的一个本征向量，且 |φ> 和 |x> 对应同一个本征值
4. 哈密顿量演化模拟从 H0 逐渐演化到 H1，该过程即将初态 |φ> 演化为期望的末态 |x>
5. RM 方法的特征在于演化路径是有随机性的 (利用量子芝诺效应QZE)
  - 多步迭代演化，步数 q 由 矩阵A的条件数κ 和 容许误差ε 决定下界
  - 每一步将含时哈密顿量 H(s) 作用在系统上演化 Tj 物理时长, Tj 由随机采样而来
6. 算法2相比算法1只是降低了物理时长Tj，没有降低逻辑步数q
  - ~300步迭代，精度0.97~0.99 ; ~3000步迭代，精度0.999
  - 时长 Tj 采样范围基本上在 [0, 10]之间
'''
def arXiv_1805_10549_RM():
  # Eq. 3 definitions
  # A(s) := (1 - s) * Z ⊗ I + s * X ⊗ A
  nq = int(np.log2(N))
  A_s = lambda s: (1 - s) * np.kron(σz, np.eye(2**nq)) + s * np.kron(σx, A)
  # |\bar{b}> := |+,b>
  b_bar: ndarray = np.kron(h0, b)
  # P_{\bar{b}}^⊥ := I - |\bar{b}><\bar{b}|
  P = np.eye(2**(1+nq)) -  b_bar @ b_bar.conj().T
  # H(s) := A(s) P_{\bar{b}}^⊥ A(s)
  H_s = lambda s: A_s(s) @ P @ A_s(s)
  # Eq. 4
  def x_s(s:float) -> ndarray:
    nonlocal A_s, b_bar
    qs = np.linalg.inv(A_s(s)) @ b_bar
    return qs / np.linalg.norm(qs)
  # H(s)|x(s)> = 0  
  for s in np.linspace(0.0, 1.0, 1000):
    assert np.allclose(H_s(s) @ x_s(s), np.zeros(2**(1+nq)))
  # As s goes 0 -> 1, |x(0)> = |-,b> evolves to |x(1)> = |+,x>
  assert np.allclose(x_s(0), np.kron(h1, b))
  assert np.allclose(x_s(1), np.kron(h0, x))
  print('ideal init state:',  x_s(0).T[0].round(4))
  print('ideal final state:', x_s(1).T[0].round(4))

  if not 'simplified version: the ancilla bit is omittable when A is posdef':
    assert is_posdef(A), 'A is not positive-definite'
    A_s = lambda s: (1 - s) * np.eye(2**nq) + s * A
    b_bar: ndarray = b
    P = np.eye(2**nq) -  b_bar @ b_bar.conj().T
    H_s = lambda s: A_s(s) @ P @ A_s(s)
    def x_s(s:float) -> ndarray:
      nonlocal A_s, b_bar
      qs = np.linalg.inv(A_s(s)) @ b_bar
      return qs / np.linalg.norm(qs)
    for s in np.linspace(0.0, 1.0, 1000):
      assert np.allclose(H_s(s) @ x_s(s), np.zeros(2**(nq)))
    assert np.allclose(x_s(0), b)
    assert np.allclose(x_s(1), x)

  # Eq. 5~7
  def s_v(v:float) -> float:
    t = np.sqrt(1 + κ**2) / (np.sqrt(2) * κ)
    p = np.exp(v * t) + 2 * κ**2  - κ**2 * np.exp(-v * t)
    q = 2 * (1 + κ**2)
    return p / q
  def v_lim() -> Tuple[float, float]:
    t = np.sqrt(1 + κ**2) / (np.sqrt(2) * κ)
    v_a = (1 / t) * np.log(κ * np.sqrt(1 + κ**2) - κ**2)
    v_b = (1 / t) * np.log(    np.sqrt(1 + κ**2) + 1)
    return v_a, v_b
  v_a, v_b = v_lim()
  print('v_a:', v_a)
  print('v_b:', v_b)
  # Eq. 8
  q_ref = np.log(κ)**2 / ε    # step count, 310.7277599582784
  q = int(q_ref)
  print(f'q: {q} (ref: {q_ref})')
  v_j = np.linspace(v_a, v_b, q+2)    # [v_a, v1, ... vq, ... v_b]
  assert np.isclose(s_v(v_a), 0.0)
  assert np.isclose(s_v(v_b), 1.0)
  print()

  # arXiv:1311.7073
  # "Δ is the spectral gap of H, that is, the smallest (absolute) difference
  # between the eigenvalue of |ψ(s)> and any other eigen value."
  def Δ_star(s:float) -> float:
    # NOTE: 这里精确计算了含时哈密顿量在给定时间下的 Δ，这在实际应用中是不可能的
    # 实际应结合 H0 和 H1 及中间若干采样来估算一个 Δ 上界，这可能也是论文中该符号带星号 Δ* 的原因
    nonlocal H_s
    evs = np.linalg.eigvals(H_s(s))
    return abs(evs[0] - evs[1])

  # Algo. 1
  print('>> [Algo 1] O(κ^2*log(κ)/ε)')
  # init state: |x(0)> = |-,b>
  qs = np.kron(h1, b)
  print('init state:', qs.T[0].round(4))
  for j in range(1, q+1):
    s_j = s_v(v_j[j])
    t_j = np.random.uniform(low=0, high=2*np.pi/Δ_star(s_j))
    U_H = expm(-1j*t_j*H_s(s_j))
    qs = U_H @ qs
  print('final state:', qs.T[0].round(4))
  x_ref = x_s(1)
  fidelity: ndarray = x_ref.conj().T @ qs
  print('fidelity:', fidelity.item())

  # Eq. 10
  # H'(s) := σ+ ⊗ (A(s) P_{\bar{b}}^⊥) + σ- ⊗ (P_{\bar{b}}^⊥ A(s))
  # As s goes 0 -> 1, |0,x(0)> evolves to |0,x(1)>
  H_hat_s = lambda s: np.kron(σp, A_s(s) @ P) + np.kron(σm, P @ A_s(s))
  # <0|<x(s)| H'(s) |1>|\bar{b}> = 0
  # "Hamiltonian does not allow for transitions between the two eigenstates"
  for s in np.linspace(0.0, 1.0, 1000):
    assert np.isclose(np.kron(v0.conj().T, x_s(s).conj().T) @ H_hat_s(s) @ np.kron(v1, b_bar), 0)

  print('-' * 42)

  # Algo. 2
  print('>> [Algo 2] O(κ*log(κ)/ε)')
  # init state: |0>|x(0)>
  qs = np.kron(v0, x_s(0))
  print('init state:', qs.T[0].round(4))
  for j in range(1, q+1):
    s_j = s_v(v_j[j])
    t_j = np.random.uniform(low=0, high=2*np.pi/np.sqrt(Δ_star(s_j)))
    U_H = expm(-1j*t_j*H_hat_s(s_j))
    qs = U_H @ qs
  print('final state:', qs.T[0].round(4))
  x_ref = np.kron(v0, x_s(1))
  fidelity = x_ref.conj().T @ qs
  print('fidelity:', fidelity.item())

print('=' * 42)
print('[arXiv_1805_10549_RM]')
arXiv_1805_10549_RM()
print()


'''
[Adiabatic Quantum Computing]
1. 准备两个哈密顿量 H0 和 H1
2. 将线性系统 Ax=b 的解 x 编码为 H1 的一个零空间向量 |x>
3. 制备系统初态 |φ> 为 H0 的一个零空间向量，且 |φ> 和 |x> 对应同一个索引
4. 哈密顿量演化模拟从 H0 逐渐演化到 H1，该过程即将初态 |φ> 演化为期望的末态 |x> (相差一个全局相位)
5. 本文提出的 AQC 主要是改进含时哈密顿量的调度函数 f(s)，相比上文 RM 方法
  - 迭代步数依然没有降低 (~1k)
  - vanilla_AQC 所要求的物理时间 Tj 相当大 (~2w)，并且 T 的调参有些困难
  - AQC(P=2) 时间复杂度在 κ 和 ε 方面与 RM(2) 持平
  - AQC(EXP) 需要在 ε < 1e-5 量级才比 AQC(P)有时间优势，但精度根本不行 (可能因为要求 A 必须正定)
'''
def arXiv_1909_05500_AQC():
  # simplified version in arXiv:1805.10549 (when A is posdef?)
  Qb = np.eye(N) - b @ b.conj().T
  H0 = np.kron(σx, Qb)
  H1 = np.kron(σp, A @ Qb) + np.kron(σm, Qb @ A)
  print_matrix(H0, 'H0')
  print_matrix(H1, 'H1')

  # https://byjus.com/question-answer/what-is-nullspace-of-a-matrix/
  H0_nullspace = [    # H0 @ null(H0) = 0
    np.kron(v0, b),   # \tilde{b}
    np.kron(v1, b),   # \bar{b}
  ]
  H1_nullspace = [    # H1 @ null(H1) = 0
    np.kron(v0, x),   # \tilde{x}; if A|x> ~ |b>, then Qb*A|x> = Qb*|b> = 0
    np.kron(v1, b),   # \bar{b}
  ]
  x_ref = H1_nullspace[0]
  print('final state (ideal):', x_ref.T[0].round(4))
  print()

  def run_with_sched_func(f:Callable[[float], float], sched:str='linear'):
    # Let f: [0, 1] -> [0, 1] be a scheduling function, we'll have the AQC evolution:
    #   H(f(s)) = (1 - f(s)) * H0 + f(s) * H1
    H_s = lambda s: (1 - f(s)) * H0 + f(s) * H1
    # and the null-space for H(f(s)) is:
    #   H1_nullspace = [
    #     np.kron(v0, x(s)),   # \tilde{x}(s)
    #     np.kron(v1, b),      # \bar{b}
    #   ]
    # where |\tilde{x}(s=0)> = |\tilde{b}> and |\tilde{x}(s=1)> = |\tilde{x}> for **any** s
    # hence |\tilde{x}(s)> is the **desired adiabatic path**

    # 制备初态: |\tilde{x}(0)> = |\tilde{b}> = |0,b>
    qs = H0_nullspace[0]
    print('init state:', qs.T[0].round(4))
    # 含时演化
    if sched == 'linear':   # at the end of Chap. 2
      T_ref = κ**3 / ε
    elif sched == 'poly':   # near Eq. 7
      T_ref = κ * np.log(κ) / ε
    elif sched == 'exp':    # near Eq. 9
      T_ref = κ * np.log(κ)**2 * np.log(np.log(κ) / ε)**4
    T = int(T_ref)          # 每一步演化的物理时间 (这是一个神秘的超参，过大过小都会直接结果不对！！)
    print(f'T: {T} (ref: {T_ref})')
    M = 1000                # 手工指定迭代次数 ~O(κ^2)，越大精度越高
    h = 1 / M               # 用于放缩 T 的步长 (这是一个神秘的超参，过大过小都会直接结果不对！！)
    for m in range(1, M+1):
      sm = m * h
      U_H = expm(-1j*T*h*H_s(sm))
      qs = U_H @ qs
    # 读出末态: |ψ_T(1)> = |\tilde{x}>, 解出 |x>
    print('final state:', qs.T[0].round(4))
    fidelity = x_ref.conj().T @ qs
    print('fidelity:', fidelity.item())

  print('[vanilla_AQC] O(κ^3/ε)')
  f = lambda s: s
  run_with_sched_func(f, 'linear')
  print('-' * 42)

  # f is ROC-curve-like
  print('[AQC(P)] O(κ/ε) ~ O(κ*log(κ)/ε)')
  def f_(p:float, s:float) -> float:
    t = 1 + s * (κ**(p-1) - 1)
    return κ / (κ - 1) * (1 - t**(1 / (1 - p)))
  run_with_sched_func(partial(f_, 1.001), 'poly')
  run_with_sched_func(partial(f_, 1.25),  'poly')
  run_with_sched_func(partial(f_, 1.5),   'poly')
  run_with_sched_func(partial(f_, 1.75),  'poly')
  run_with_sched_func(partial(f_, 2),     'poly')
  print('-' * 42)

  # f is sigmoid-like
  print('[AQC(EXP)] O(κ*log^2(κ)*log^4(log(κ)/ε))')
  def intg(s_lim:float, ds:float=1e-2):
    sum = 0.0
    s = ds
    while s < s_lim:
      sum += np.exp(-1 / (s * (1 - s))) * ds
      s += ds
    return sum
  c_e = intg(1)
  f = lambda s: intg(s) / c_e
  run_with_sched_func(f, 'exp')

print('=' * 42)
print('[arXiv_1909_05500_AQC]')
arXiv_1909_05500_AQC()
print()
