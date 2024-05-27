#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/21

# 符合本赛题限制和约束的离散绝热线性求解器设计 🎉

'''
第二问的比赛要求有如下限制或提示：
  - 代码中 Block-Encoding 的实现与第一问中一致，使用 QPanda 的 matrix_decompose() 接口
  - 参考文献 arXiv:1909.05500 中的调度函数 f(s) 和含时哈密顿量 H(s) 的构造方法，选取 f(s) = s, s ∈ [0, 1], Δs = 1/200
  - 使用一阶近似 exp(-iH) ≈ I - iH
  - 基于 QPanda 实现离散绝热线性求解器，输出 |x> 和 <x_r|x>
这使得好像出题方看起来并不是要求选手去实现论文中的那些经典方法，我们根据这些限制来重新评估所暗示的解决方案!!

- 哈密顿量模拟[1,2]: 若系统的哈密顿量为不含时的 H，则初态 |ψ(0)> 经过时间 t 后会自然演化为 |ψ(t)>
  - |ψ(t)> = exp(-iHt) |ψ(0)>
  - Pauli 算符的指数化线路[1] (根据矩阵形式硬凑出来的)
    - exp(-iIt) = Rz(-t) X Rz(-t) X
    - exp(-iXt) = H Rz(-t) X Rz(t) X H
    - exp(-iYt) = Rx(pi/2) X Rz(-t) X Rz(t) Rx(-pi/2)
    - exp(-iZt) = X Rz(-t) X Rz(t)
- 绝热演化[3,4]: 系统的哈密顿量从 H0 缓慢演化到 H1，若系统初态恰为 H0 的第k本征态，则最终会演化到 H1 的第k本征态
  - 系统的含时哈密顿量可近似为线性插值: H(s) = (1 - s) * H0 + s * H1, s: 0 -> 1
  - 离散绝热演化: Trotter 分解可以把哈密顿量的和 exp(-iT(H0+H1)) 分解成小片的积 Πt exp(-i(t/T)H0)exp(-i(t/T)H1)
ref: 
- [1] https://zhuanlan.zhihu.com/p/150292241
- [2] https://zhuanlan.zhihu.com/p/529720555
- [3] https://zhuanlan.zhihu.com/p/24086259
- [4] https://www.cnblogs.com/dechinphy/p/annealer.html
根据以上不难得出解决方案：
  0. 规范化所求线性方程组 A * x = b 右端
  1. 将系统初态制备为 |b>
  2. 进行从 H0 到 H1 的离散绝热演化
    - 精确设计哈密顿量 H0 和 H1，使得 |b> 是 H0 的一个本征态/零空间
    - 确定调度函数 f(s) = s, 构造含时哈密顿量 H(s) = (1 - s) * H0 + s * H1
    - 进行 S=200 步的演化，第 s 步使用哈密顿量 H(s) 演化恰当时长 T=1 (?)
      - 标准方案即用酉矩阵 exp(-iH(s)t) 进行演化 (QPanda 提供了用于直接数学模拟的 expMat，但未提供 TimeEvolution 量子线路)
      - 但按题目要求，需要使用一阶近似 TE = exp(-iHt) ≈ I - iHt = \hat{TE} 取代该酉矩阵进行演化
      - 由于近似矩阵 \hat{TE} 非酉，需要进一步使用其 BlockEncoding 形式来进行演化；选型思考最后决定用 QSVT 作答
        - QSVT 方案不保证任意输入都可用多项式个门高效地实现，比赛要求"可用量子线路实现" (但好像没要求高效?)
        - LCU 方案暂时不支持复数矩阵 (QPanda 接口暂时不能分解 Pauli-Y)
        - ARCSIN/FABLE 方案要求矩阵元素 |a_{i,j}| \leq 1，该方案大概也是比赛所暗示的标准解答思路，但 \hat{TE} 值域是无界的
  3. 读出末态 |x>
  4. 计算保真度
'''

from functools import partial
from tqdm import tqdm
from scipy.linalg import expm
import pennylane as qml
from pennylane.measurements import StateMP

from utils import *

def exp_iHt_approx(H:ndarray, t:float=1) -> ndarray:
  # exp(-iHt) = I - iHt
  return np.eye(H.shape[0]) - 1j * H * t

def block_encode(A:ndarray) -> ndarray:
  N = A.shape[0]
  nq = int(np.ceil(np.log2(N)))
  n_wires = nq + 1

  dev = qml.device('default.qubit', wires=n_wires)
  @qml.qnode(dev)
  def block_encode_method(A:ndarray) -> StateMP:
    qml.BlockEncode(A, wires=range(n_wires))
    return qml.state()

  return qml.matrix(block_encode_method)(A)


# hprams
H_s_method = 'AQC'
f_s_method = 'linear'
f_s_p = 2.0
use_approx = True
# NOTE: 这里 S*T 是真实物理时长，要保证其积足够大 (缓慢演化)
# 且条件允许的情况下，演化步数 S 应该尽可能大，而每步演化时间 T 应该尽可能小
S = 200   # 演化步数 (根据赛题要求固定)
T = 10    # 每步演化时长

if H_s_method == 'RM':
  A_s = lambda s: (1 - s) * np.kron(σz, I_(nq)) + s * np.kron(σx, A)
  b_bar: ndarray = np.kron(h0, b)
  P = I_(1 + nq) -  b_bar @ b_bar.conj().T
  H_s = lambda s: A_s(s) @ P @ A_s(s)
  f_s_method = None
  init_qs = np.kron(h1, b)
  final_qs = np.kron(h0, x)
elif H_s_method == 'RMs':    # simple version, when A is posdef
  A_s = lambda s: (1 - s) * I_(nq) + s * A
  P = I_(nq) -  b @ b.conj().T
  H_s = lambda s: A_s(s) @ P @ A_s(s)
  f_s_method = None
  init_qs = b
  final_qs = x
elif H_s_method == 'AQC':
  Qb = I_(nq) - b @ b.conj().T
  H0 = np.kron(σx, Qb)
  H1 = np.kron(σp, A @ Qb) + np.kron(σm, Qb @ A)
  H_s = lambda s: (1 - f(s)) * H0 + f(s) * H1
  init_qs = np.kron(v0, b)
  final_qs = np.kron(v0, x)
else: raise ValueError

if f_s_method == 'linear':
  f = lambda s: s
elif f_s_method == 'poly':
  def f_(p:float, s:float) -> float:
    t = 1 + s * (κ**(p-1) - 1)
    return κ / (κ - 1) * (1 - t**(1 / (1 - p)))
  f = partial(f_, f_s_p)
else: pass

if use_approx:
  n_be_ancilla = 1
else:
  n_be_ancilla = 0


def run(S:int, T:int, log:bool=True) -> float:
  # 块编码辅助比特空间扩张
  anc0 = v0
  for i in range(n_be_ancilla - 1):
    anc0 = np.kron(anc0, v0)
  # 制备初态
  qs = np.kron(anc0, init_qs) if use_approx else init_qs
  if log: print('init state:', qs.T[0].round(4))
  # 绝热演化
  for s in range(S):
    H = H_s(s / S)  # NOTE: 此处直接模拟哈密顿量的和，暂不用 trotter 分解
    if use_approx:
      exp_iHt = exp_iHt_approx(H, T)
      U_iHt = block_encode(exp_iHt)
    else:
      U_iHt = expm(-1j*H*T)
    qs = U_iHt @ qs
  # 读取末态
  if log: print('final state:', qs.T[0].round(4))
  x_ref = np.kron(anc0, final_qs) if use_approx else final_qs
  if log: print('ref state:', x_ref.T[0].round(4))
  fidelity = x_ref.conj().T @ qs
  if log: print('fidelity:', fidelity.item())
  return abs(fidelity.item())


def run_grid():
  T_list = list(range(1, 32))
  S_list = [100 * i for i in range(1, 5)]
  res = np.zeros([len(T_list), len(S_list)])
  pbar = tqdm(total=len(T_list)*len(S_list))
  for i, T in enumerate(T_list):
    for j, S in enumerate(S_list):
      res[i, j] = run(S, T, log=False)
      pbar.update(1)
  pbar.close()

  f_s_method_sfx = f'({f_s_p})' if f_s_method == 'poly' else ''
  name_suffixes = [
    f'_H={H_s_method}'if H_s_method else '',
    f'_f={f_s_method}{f_s_method_sfx}'if f_s_method else '',
    f'_approx'if use_approx else '',
  ]
  name = 'sim' + ''.join(name_suffixes)

  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure(figsize=(6, 12))
  sns.heatmap(res, cbar=True, annot=True, fmt='.4f')
  plt.ylabel('T - evolution time')
  plt.xlabel('S - n_iter')
  plt.yticks(list(range(len(T_list))), T_list)
  plt.xticks(list(range(len(S_list))), S_list)
  plt.suptitle(name)
  plt.gca().invert_yaxis()
  plt.tight_layout()
  fp = IMG_PATH / f'{name}.png'
  print(f'>> savefig to {fp}')
  plt.savefig(fp, dpi=800)


#run_grid()
run(S, T)
