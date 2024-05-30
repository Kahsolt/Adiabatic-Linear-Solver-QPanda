#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/15

# 第二问: 线性方程组问题 A * x = b 转 绝热演化问题 H(s) = (1 - f(s)) * H0 + f(s) * H1
# arXiv:1805.10549: RM-based; 非常好论文，使我光速复现！！
# arXiv:1909.05500: AQC(p), AQC(exp), QAOA; 复现了AQC方法部分，论文公式有小错误
# arXiv:1910.14596: Eigen-Filter
# arXiv:2111.08152 Chap. IV: QDA, Walk operator, Block Encoding H(s)

from typing import Callable

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
  - |φ> = |x(0)> = |\hat{b}> = |-,b>
  - |x> = |x(1)> = |\hat{x}> = |+,x>
  - |\hat{x_s}> = Π_j exp(-i*T_j*H(s_j)) |\hat{b}>
4. 哈密顿量演化模拟从 H0 逐渐演化到 H1，该过程即将初态 |φ> 演化为期望的末态 |x>
  - 含时哈密顿量 H(s) 被拆成小片，且指数化成酉矩阵/量子门之后，作用在量子态上进行演化
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
  A_s = lambda s: (1 - s) * np.kron(σz, I_(nq)) + s * np.kron(σx, A)
  # |\bar{b}> := |+,b>
  b_bar: ndarray = np.kron(h0, b)
  # P_{\bar{b}}^⊥ := I - |\bar{b}><\bar{b}|
  P = I_(1+nq) -  b_bar @ b_bar.conj().T
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
    A_s = lambda s: (1 - s) * I_(nq) + s * A
    b_bar: ndarray = b
    P = I_(nq) -  b_bar @ b_bar.conj().T
    H_s = lambda s: A_s(s) @ P @ A_s(s)
    def x_s(s:float) -> ndarray:
      nonlocal A_s, b_bar
      qs = np.linalg.inv(A_s(s)) @ b_bar
      return qs / np.linalg.norm(qs)
    for s in np.linspace(0.0, 1.0, 1000):
      assert np.allclose(H_s(s) @ x_s(s), np.zeros(2**(nq)))
    assert np.allclose(x_s(0), b)
    assert np.allclose(x_s(1), x)

  # Eq. 5~7, schedule for s_j = f_j(j) for j in vrng [1, q]
  q, f_j = make_f_s_RM(debug=True)

  # arXiv:1311.7073
  # "Δ is the spectral gap of H, that is, the smallest (absolute) difference
  # between the eigenvalue of |ψ(s)> and any other eigen value."
  def Δ_s(s:float) -> float:      # 含时哈密顿量在给定时间下的精确 Δ
    nonlocal H_s
    evs = np.linalg.eigvals(H_s(s))
    return abs(evs[0] - evs[1])
  def Δ_star(s:float) -> float:   # 含时哈密顿量的 Δ 下界
    f = lambda s: s
    # arXiv:1909.05500 Eq. 1
    return 1 - f(s) + f(s) / κ

  # Algo. 1
  print('>> [Algo 1] O(κ^2*log(κ)/ε)')
  # init state: |x(0)> = |-,b>
  qs = np.kron(h1, b)
  print('init state:', state_vec(qs))
  for j in range(1, q+1):
    s_j = f_j(j)
    t_j = np.random.uniform(low=0, high=2*np.pi/Δ_star(s_j))
    U_H = expm(-1j*t_j*H_s(s_j))
    qs = U_H @ qs
  print('final state:', state_vec(qs))
  x_ref = x_s(1)
  print('fidelity:', get_fidelity(qs, x_ref))

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
  print('init state:', state_vec(qs))
  for j in range(1, q+1):
    s_j = f_j(j)
    t_j = np.random.uniform(low=0, high=2*np.pi/np.sqrt(Δ_star(s_j)))
    U_H = expm(-1j*t_j*H_hat_s(s_j))
    qs = U_H @ qs
  print('final state:', state_vec(qs))
  x_ref = np.kron(v0, x_s(1))
  print('fidelity:', get_fidelity(qs, x_ref))

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
  Qb = I_(nq) - b @ b.conj().T
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
  print('final state (ideal):', state_vec(x_ref))
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

    # 演化参数
    if sched == 'linear':   # at the end of Chap. 2
      T_ref = κ**3 / ε
    elif sched == 'poly':   # near Eq. 7
      T_ref = κ * np.log(κ) / ε
    elif sched == 'exp':    # near Eq. 9
      T_ref = κ * np.log(κ)**2 * np.log(np.log(κ) / ε)**4
    T = int(T_ref)          # 总演化的物理时间 (这是一个神秘的超参，过大过小都会直接结果不对！！)，报告视频消融实验里最高取到 T = 1000
    print(f'T: {T} (ref: {T_ref})')
    S = 1000                # 手工指定迭代次数 ~O(κ^2)，越大精度越高
    h = 1 / S               # 每步演化的物理时间 (这是一个神秘的超参，过大过小都会直接结果不对！！)
    # 制备初态: |\tilde{x}(0)> = |\tilde{b}> = |0,b>
    qs = H0_nullspace[0]
    print('init state:', state_vec(qs))
    # 含时演化
    for s in range(1, 1+S):
      H = H_s(s / S)        # NOTE: 此处直接模拟哈密顿量的和，暂不用 trotter 分解
      U_iHt = expm(-1j*H*(T*h))
      qs = U_iHt @ qs
    # 读出末态: |ψ_T(1)> = |\tilde{x}>, 解出 |x>
    print('final state:', state_vec(qs))
    print('fidelity:', get_fidelity(qs, x_ref))

  print('[vanilla_AQC] O(κ^3/ε)')
  run_with_sched_func(make_f_s_linear(), 'linear')
  print('-' * 42)

  # f is ROC-curve-like
  print('[AQC(P)] O(κ/ε) ~ O(κ*log(κ)/ε)')
  run_with_sched_func(make_f_s_AQC_P(1.001), 'poly')
  run_with_sched_func(make_f_s_AQC_P(1.25),  'poly')
  run_with_sched_func(make_f_s_AQC_P(1.5),   'poly')
  run_with_sched_func(make_f_s_AQC_P(1.75),  'poly')
  run_with_sched_func(make_f_s_AQC_P(2),     'poly')
  print('-' * 42)

  # f is sigmoid-like
  print('[AQC(EXP)] O(κ*log^2(κ)*log^4(log(κ)/ε))')
  run_with_sched_func(make_f_s_AQC_EXP(), 'exp')

print('=' * 42)
print('[arXiv_1909_05500_AQC]')
arXiv_1909_05500_AQC()
print()


'''
[Eigenvalue Filtering]
1. 先用 AQC(p) 跑出来一个近似解 |\hat{x0}> = \gamma_0 |0,x> + \gamma_1 |1,b> + |⊥>
2. 纯线性代数的思路，乘上一个矩阵进行过滤 R_l(H1/d; 1/(dk)) |\hat{x0}> 将会去除无关成分 |⊥>
  - R_l(x,Δ) 是一个标量函数，如何定义它的矩阵版本 R_l(H,Δ)? element-wise??
  - |x> ~= P_{λ=0}(H1)|x0>
3. 对第一辅助比特进行测量，将有 \gamma_0^2 概率去除误差成分 |1,b> 而得到目标 |0,x>
'''
def arXiv_1910_14596_EF():
  print('[EF(AQC)] O(κlog(1/ε))')
  # https://zh.wikipedia.org/wiki/切比雪夫多项式
  def T_l(x:float, l:float=16) -> float:
    if -1 < x < 1: return np.cos(l * np.arccos(x))
    if x > 1:      return np.cosh(l * np.arccosh(x))
    if x < -1:     return (-1)**l * np.cosh(l * np.arccosh(-x))
  def R_l(x:float, l:float=16, Δ:float=0.1) -> float:
    t = lambda x: (x**2 - Δ**2) / (1 - Δ**2)
    p = T_l(-1 + 2*(t(x)), l)
    q = T_l(-1 + 2*(t(0)), l)
    return p / q

  print('[EF(QZE)] O(κlog(1/ε))')
  # Eq. 14
  f = lambda s: (1 - κ**(-s)) / (1 - κ**(-1))

# FIXME: NotImplemented
#print('=' * 42)
#print('[arXiv_1910_14596_EF]')
#arXiv_1910_14596_EF()
#print()


'''
[Quantum Discrete Adiabatic]
1. 纯线性代数的思路，先 QWalk 到一个近似解，再 Eigen-Filter with LCU
2. QWalk 算子 W_T(s) 的实际效果似乎是 expm(i * arcsin(H) * λ)
'''
def arXiv_2111_08152_QDA():
  # Eq.114, the AQC(P) borrowed from arXiv:1909.05500
  f = make_f_s_AQC_P(1.5)

  if not 'Appendix E':
    # NOTE: 以下为附录 E 的内容，描述了系数矩阵 A 是以块编码的形式 U_A 给出时，如何进一步块编码 H(s)
    # 但赛题里的 A 是直接给出矩阵形式的，所以可以直接按公式 Eq.124 来构造矩阵进而整体编码 H(s)，完全不用看这坨答辩 :(
    # arXiv:2312.07690 Fig. 4 给出了 H(s) 块编码的线路实现，并再度重申了：
    # "The walk operator W_T(s) can be completed by a reflection about zero on all ancilla registers except a_h"
    # 但确定这个 reflection 操作依然比较困难，即使 pennylane 中有相关实现

    # tmp vars
    nq = int(np.log2(A.shape[0]))
    v0v0 = v0 @ v0.conj().T
    v0v1 = v0 @ v1.conj().T
    v1v0 = v1 @ v0.conj().T
    v1v1 = v1 @ v1.conj().T

    # Eq. E1 [D=8]
    from pennylane import qml
    dev = qml.device('default.qubit', wires=3)
    @qml.qnode(dev)
    def fable_method(A:ndarray):
      qml.FABLE(A, wires=range(3), tol=0)
      return qml.state()
    U_A: ndarray = qml.matrix(fable_method)(A).real   # D=8
    nq_BE = int(np.log2(U_A.shape[0]))
    assert np.allclose(U_A[:N, :N]*N, A), 'block encode check error'
    # Eq. E2 (D=2)
    U_b = RY(2*np.arccos(b[0, 0]))  # U_b|0> = |b>
    # Eq. E3 (D=4), {a1,A}
    A_f = lambda s: (1 - f(s)) * np.kron(σz, I_(nq)) + f(s) * (np.kron(v0v1, A) + np.kron(v1v0, A.conj().T))
    # Eq. E4 (D=32) {a2,a1,A,BE_a1,BE_a2}, A and b is the same qubit
    def U_A_f(s:float) -> ndarray:
      p = (1 - f(s)) * np.kron(v0v0,  np.kron(σz, I_(nq_BE)))
      q =      f(s)  * np.kron(v1v1, (np.kron(v0v1, U_A) + np.kron(v1v0, U_A.conj().T)))
      return p + q
    # Eq. E5 (D=4) {a1,b}
    Q_b = I_(1+nq) - np.kron(v1v1, b @ b.conj().T)
    # Eq. E7 (D=8) {a3,a1,b}
    t1 = np.kron(I_(2), U_b.conj().T)
    t2 = np.kron(v0v0, I_(1+nq)) + np.kron(v1v1, 2*I_(1+nq) - np.kron(v1v1, v0v0))
    t3 = np.kron(I_(2), U_b)
    U_Q_b = t1 @ t2 @ t3
    # Eq. E8 (D=8) {a4,(a1,A)*(a1,b)}
    H_s = lambda s: np.kron(v0v1, A_f(s) @ Q_b) + np.kron(v1v0, Q_b @ A_f(s))
    # Eq. E9 (D=16) {a4,a3,a1,b}
    C_U_Q_b_1 = np.kron(v0v0, I_(2+nq)) + np.kron(v1v1, U_Q_b)
    C_U_Q_b_0 = np.kron(v1v1, I_(2+nq)) + np.kron(v0v0, U_Q_b)
    # Eq. 127 (D=2)
    R_s = lambda s: np.asarray([
      [1 - f(s), f(s)],
      [f(s), f(s) - 1],
    ]) / np.sqrt((1 - f(s))**2 + f(s)**2)
    # Eq. E10 & E11 (D=4) {a4,a2}
    CR_s_0 = lambda s: np.kron(v0v0, R_s(s)) + np.kron(v1v1, H)
    CR_s_1 = lambda s: np.kron(v1v1, R_s(s)) + np.kron(v0v0, H)
    # Eq. E12 (need 7 qubits in total)
    def U_H_s(s:float) -> ndarray:
      ''' ubit order: |a4,a3,a2,a1,ψ,BE_a1,BE_a2>'''
      nq_circ = 4 + nq_BE
      g = I_(nq_circ)       # start from I
      g = g @ np.kron(I_(1), np.kron(H, I_(nq_circ-2)))
      g = g @ C_U_Q_b_1     # a4,a3,a1,b
      g = g @ CR_s_0(s)     # a4,a2
      g = g @ np.kron(I_(2), U_A_f(s))    # {a2,a1,A,BE_a1,BE_a2}
      g = g @ CR_s_1(s)     # a4,a2
      g = g @ C_U_Q_b_0     # a4,a3,a1,b
      g = g @ np.kron(I_(1), np.kron(H, I_(nq_circ-2)))
      g = g @ np.kron(σx, I_(nq_circ-1))
      return g
    # "随机游走算子是 H(s) 的块编码围绕着零向量在辅助比特上的反射", wtf??
    def W_T(s:float) -> ndarray:
      H_s = U_H_s(s)
      H_s_dagger = H_s.conj().T
      return 'wtf'

  print('[QDA] O(κ*log(1/ε)')
  # 制备初态: |0,+,b> of H0
  qs = np.kron(v0, np.kron(h0, b))
  # 随机游走
  T_ref = κ * np.log(1 / ε)
  T = int(T_ref)    # T = 5 * 10**4   # Chap. IV, Sect. E
  print(f'T: {T} (ref: {T_ref})')
  # Eq. 130 & 131
  c_1_s = lambda s: 2 * T * (f(s + 1 / T) - f(s))
  def c_2_s(s:float) -> float:
    f_p  = lambda s, ds=0.001: (f  (s + ds) - f  (s)) / ds   # f 的一阶/二阶导
    f_pp = lambda s, ds=0.001: (f_p(s + ds) - f_p(s)) / ds
    trails = [s, s+1/T] + ([] if np.isclose(s, 1 - 1 / T) else [s+2/T])
    return 2 * max([2 * np.abs(f_p(τ))**2 + np.abs(f_pp(τ))] for τ in trails)
  # 走！
  for t in range(1, T+1):
    s = t / T
    qs = W_T(s) @ qs
  # 读出末态: |ψ_T(1)> = |\tilde{x}>, 解出 |x>
  print('final state:', state_vec(qs))
  x_ref = np.kron(v1, np.kron(h1, x))
  print('fidelity:', get_fidelity(qs, x_ref))

# FIXME: NotImplemented
#print('=' * 42)
#print('[arXiv_2111_08152_QDA]')
#arXiv_2111_08152_QDA()
#print()


'''
[Efficient Quantum Linear Solver with Detailed Running Costs]
1. 纯粹破事水的思路，先 RM 到一个近似解，再 Eigen-Filter
'''
def arXiv_2305_11352_EQLS():
  pass

# FIXME: NotImplemented
#print('=' * 42)
#print('[arXiv_2305_11352_EQLS]')
#arXiv_2305_11352_EQLS()
#print()
