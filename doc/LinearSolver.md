# qda_linear_solver 模块文档

ℹ 我们基于 QPanda (C++) 实现了 3 个线性求解器函数

```cpp
// 理想的绝热演化求解器
VectorXcd linear_solver_ideal(MatrixXcd A, VectorXcd b);
// 严格符合赛题诸多要求限制的求解器
VectorXcd linear_solver_contest(MatrixXcd A, VectorXcd b);
// 为精度做了各种优化的求解器
VectorXcd linear_solver_ours(MatrixXcd A, VectorXcd b);
```

### ideal

`linear_solver_ideal` 实现了基于理想哈密顿模拟的绝热演化求解器，其中时间演化算子 $ e^{-iHt} $ 直接由数学计算得到，通过 `matrix_decompose()` 来获取对应的量子逻辑门线路实现。基本程序结构如下：

```cpp{.line-numbers}
// 绝热演化参数
const int S = 200;    // 总演化阶段步数
const int T = 10;     // 单步哈密顿量模拟时间
// 制备系统初态 |b>
qcir << amplitude_encode(qv, amplitude);
// 制备含时哈密顿量 H_s
H_s = (1 - s) * H0 + s * H1
// 绝热演化
for (int s = 1; s <= S; s++) {
  // 含时哈密顿量近似为不含时
  MatrixXcd H = H_s(float(s) / S);
  // 时间演化算子的矩阵形式
  MatrixXcd iHt = dcomplex(0, -1) * H * T;
  MatrixXcd U_iHt = iHt.exp();
  // 矩阵形式转化为量子逻辑门线路
  qcir << matrix_decompose(U_iHt, qv);
}
// 概率测量读取振幅，解出 |x>
QProg qprog = createEmptyQProg() << qcir;
qvm.directlyRun(qprog);
```

### contest

`linear_solver_contest` 实现了基于近似哈密顿模拟的绝热演化求解器，其中时间演化算子 $ e^{-iHt} $ 通过一阶近似为 $ e^{-iHt} \approx I - iHt $，通过 BlockEncoding 技术获得其酉矩阵版本，再通过 `matrix_decompose()` 来获取对应的量子逻辑门线路实现。基本程序结构如下，与 `linear_solver_ideal` 的主要差别是 12 ~ 16 行的近似。

```cpp{.line-numbers}
// 绝热演化参数
const int S = 200;    // 总演化阶段步数
const int T = 1;      // 单步哈密顿量模拟时间
// 制备系统初态 |b>
qcir << amplitude_encode(qv, amplitude);
// 制备含时哈密顿量 H_s
H_s = (1 - s) * H0 + s * H1
// 绝热演化
for (int s = 1; s <= S; s++) {
  // 含时哈密顿量近似为不含时
  MatrixXcd H = H_s(float(s) / S);
  // 时间演化算子的一阶近似
  MatrixXcd iHt = exp_iHt_approx(H, T);
  // 谱范数规范化 & 进行块编码
  iHt = normalize_QSVT(iHt);
  MatrixXcd U_iHt = block_encoding_QSVT(iHt);
  // 近似的时间演化算子转化为量子逻辑门线路
  qcir << matrix_decompose(U_iHt, qv);
}
// 概率测量读取振幅，解出 |x>
QProg qprog = createEmptyQProg() << qcir;
qvm.directlyRun(qprog);
```

### ours

`linear_solver_ours` 基于 `linear_solver_contest` 迭代改造而成，引入了大量 trick 来提升结果的保真度，探索绝热演化线性求解器的能力上限，主要包含下列 trick：

- 更多的演化阶段步数 S，更长的物理演化时间 T (第 2 ~ 3 行)
- AQC(P) 调度函数 (第 7 行)
- $ e^{-iHt} $ 二阶近似 (第 14 行)
- 特征滤波 EF (第 21 ~ 25 行)

```cpp{.line-numbers}
// 绝热演化参数
const int S = 300;    // 总演化阶段步数
const int T = 10;     // 单步哈密顿量模拟时间
// 制备系统初态 |b>
qcir << amplitude_encode(qv, amplitude);
// 制备含时哈密顿量 H_s，使用 AQC(P=2) 调度函数 f(s)
H_s = (1 - f(s)) * H0 + f(s) * H1
// 绝热演化
// adiabatic evolution
for (int s = 1; s <= S; s++) {
  // 含时哈密顿量近似为不含时
  MatrixXcd H = H_s(float(s) / S);
  // 时间演化算子的二阶近似
  MatrixXcd iHt = exp_iHt_approx(H, T, 2);
  // 谱范数规范化 & 进行块编码
  iHt = normalize_QSVT(iHt);
  MatrixXcd U_iHt = block_encoding_QSVT(iHt);
  // 近似的时间演化算子转化为量子逻辑门线路
  qcir << matrix_decompose(, U_iHt, qv);
}
// 制作特征滤波矩阵 & 进行块编码
MatrixXcd EF = EF_R_l(H1);
MatrixXcd U_EF = block_encoding_QSVT(EF);
// 特征滤波矩阵转化为量子逻辑门线路
qcir << matrix_decompose(U_EF, qv);
// 概率测量读取振幅，解出 |x>
QProg qprog = createEmptyQProg() << qcir;
qvm.directlyRun(qprog);
```

#### 附录

基准单元测试运行结果参考 (`T=1000`)：

![LS_unittest](../img/LS_unittest.png)

### case-study experiment

##### 对比实验

在 [playground/qda_linear_solver.py](../playground/qda_linear_solver.py) 和 [playground/vis_eigen_filter_ls.py](../playground/vis_eigen_filter_ls.py) 中我们也实现了 Randomization Method, vanillan AQC, AQC(P), AQC(EXP) 和 Eigen Filter 作为对比参考。

解示例方程组：

$$
\begin{array}{ll}
\begin{bmatrix}
  2 & 1 \\
  1 & 0 \\
\end{bmatrix} \begin{bmatrix}
  x_1 \\
  x_2 \\
\end{bmatrix} = \begin{bmatrix}
  3 \\
  1 \\
\end{bmatrix}
\end{array}
$$

| implementation | solution | fidelity | comment |
| :-: | :-: | :-: | :-: |
| LS_ideal     | [-0.722392, -0.691484] | 0.999761 | S=200, T=S*10 |
| LS_contest   | [-0.777313,  0.629114] | 0.104793 | S=200, T=S*1 is not enough |
| LS_ours      | [-0.704977, -0.70923]  | **0.999995** | S=300, T=S*10 |
| vanilla AQC  | [-0.7134, -0.7007] | 0.999959 | T=19799 |
| AQC(p=1.001) | [-0.6869, -0.7265] | 0.999437 | T=1027 |
| AQC(p=1.25)  | [-0.7232, -0.6906] | 0.999708 | T=1027 |
| AQC(p=1.5)   | [-0.7141, -0.6997] | 0.999745 | T=1027 |
| AQC(p=1.75)  | [-0.7012, -0.7128] | 0.999875 | T=1027 |
| AQC(p=2)     | [-0.7008, -0.7134] | 0.999960 | T=1027 |
| AQC(exp)     | [-0.7137, -0.7004] | 0.999949 | T=12959 |
| RM (algo 1)  | [(0.5037+0.0351j), (0.4846+0.0373j), (0.4954-0.0359j), (0.5113-0.0258j)] | 0.997524 | precision sucks |
| RM (algo 2)  | [(0.509+0j), (0.5133+0j), (0.4881+0j), (0.4885+0j), -0.0177j, 0.0046j, 0.0181j, -0.006j] | 0.999385 | precision sucks |

ℹ Note that target solution is [0.7071, 0.7071] for all AQC-based methods (up to a global phase), while [0.5, 0.5, 0.5, 0.5] for RM-based ones

##### 消融实验

TODO: C++ 反复编译实在太烦人，再说吧……

----
by Armit
2024/5/30
