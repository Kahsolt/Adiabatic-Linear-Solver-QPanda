#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Components/Operator/PauliOperator.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "block_encoding.h"
#include "utils.h"

bool check_block_encoding(block_encoding_res &res, MatrixXcd &A, float eps=1e-5) {
  auto N = A.rows(), M = A.cols();
  auto block = res.unitary.block(0, 0, N, M) / res.subfactor;
  return (A - block).cwiseAbs().maxCoeff() < eps;
}


// Block-Encoding interface
MatrixXcd block_encoding(MatrixXcd H, BlockEncodingMethod method=BlockEncodingMethod::ARCSIN) {
  if (!is_shape_pow2(H)) {  // auto expand shape
    int N = H.rows(), M = H.cols();
    int maxN = N > M ? N : M;
    maxN = int(pow(ceil(log2(maxN)), 2));
    MatrixXcd H_ex = MatrixXcd::Zero(maxN, maxN);
    H_ex.block(0, 0, N, M) = H;
    H = H_ex;
  }
  block_encoding_res res;
  switch (method) {
    case BlockEncodingMethod::QSVT:   res = block_encoding_QSVT  (H); break;
    case BlockEncodingMethod::QSVT0:  res = block_encoding_QSVT0 (H); break;
    case BlockEncodingMethod::LCU:    res = block_encoding_LCU   (H); break;
    case BlockEncodingMethod::ARCSIN: res = block_encoding_ARCSIN(H); break;
    case BlockEncodingMethod::FABLE:  res = block_encoding_FABLE (H); break;
    default: throw invalid_argument("invalid method");
  }
  return res.unitary;
}


// Block-Encoding via QSVT-like direct construction from arXiv:2203.10236 Eq. 3.4
// Accepting arbitary square or non-sqaure matrix
block_encoding_res block_encoding_QSVT(MatrixXcd A) {
  if (spectral_norm(A) > 1) {
    cout << "spectral_norm: " << spectral_norm(A) << endl;
    throw domain_error("A must satisfy ||A||2 <= 1");
  }

  // https://pennylane.ai/qml/demos/tutorial_intro_qsvt/
  int N = A.rows(), M = A.cols();
  MatrixXcd U_A(N + M, N + M);
  U_A.block(0, 0, N, M) = A;
  U_A.block(0, M, N, N) = sqrt(MatrixXcd::Identity(N, N) - A * A.adjoint());
  U_A.block(N, 0, M, M) = sqrt(MatrixXcd::Identity(M, M) - A.adjoint() * A);
  U_A.block(N, M, M, N) = -A.adjoint();
  return block_encoding_res(U_A, 1);
}


// Block-Encoding via QSVT-like direct construction for 2x2 real symmetric matrix special case, refer to arXiv:2203.10236 Eq. 3.6
// Accepting 2x2 real symmetric matrix
block_encoding_res block_encoding_QSVT0(MatrixXcd A) {
  if (!is_square(A)) throw domain_error("A is not square");
  if (A.rows() != 2) throw domain_error("A is not 2x2 shape");
  if (!is_real(A)) throw domain_error("A is not real");
  if (spectral_norm(A) > 1) throw domain_error("A must satisfy ||A||2 <= 1");
  auto p = A(0, 0), q = A(0, 1),
       r = A(1, 0), s = A(1, 1);
  const float eps = 1e-8;
  if (abs(p - s) > eps) throw domain_error("A is not symmetric along main diagonal");
  if (abs(q - r) > eps) throw domain_error("A is not symmetric along sub diagonal");

  auto a1 = p.real(), a2 = q.real();
  auto b1 = sqrt(1 - pow(a1, 2)), b2 = sqrt(1 - pow(a2, 2));
  MatrixXcd U_a = MatrixXcd(4, 4);
  U_a << a1, a2, a1, -a2,
         a2, a1, -a2, a1,
         a1, -a2, a1, a2,
         -a2, a1, a2, a1;
  MatrixXcd U_b = MatrixXcd(4, 4);
  U_b << b1, b2, b1, -b2,
         b2, b1, -b2, b1,
         b1, -b2, b1, b2,
         -b2, b1, b2, b1;
  MatrixXcd U_A(8, 8);
  U_A << U_a, -U_b,
         U_b, U_a;
  return block_encoding_res(U_A / 2, 1 / 2);
}


// Block-Encoding via linear combination of unitaries method from arXiv:1501.01715
// Accepting square matrix, decomposable to a linear combination of unitaries
block_encoding_res block_encoding_LCU(MatrixXcd A, float eps=1e-8) {
  if (A.rows() != 2) throw domain_error("A is not 2x2 shape, currently only support 2x2 matrix");
  if (!is_shape_pow2(A)) throw domain_error("A shape is not power of 2");
  if (!is_real(A)) throw domain_error("A is not real, currently only support real matrix");

  // matrix to LCU
  int N = A.rows();
  EigenMatrixX matrix = A.real();
  PauliOperator ops = matrix_decompose_hamiltonian(matrix);
  vector<float> coeffs;
  vector<string> terms;
  for (auto item : ops.data()) {
    double coeff = item.second.real();
    if (abs(coeff) < eps) continue;
    coeffs.push_back(coeff);
    terms.push_back(item.first.first.size() == 0 ? "I0" : item.first.second);
  }

  // LCU to QCircuit
  // 1. decide system size
  size_t n_qubit = ceil(log2(N));
  size_t n_unitary = coeffs.size();
  size_t n_ancilla = ceil(log2(n_unitary));
  size_t n_qubit_ex = n_qubit + n_ancilla;
  CPUQVM qvm;
  qvm.setConfigure({n_qubit_ex, n_qubit_ex});
  qvm.init();
  // |...,a1,a0,...,q1,q0>, mind the fucking order!! :(
  QVec qv_w = qvm.qAllocMany(n_qubit);
  QVec qv_a = qvm.qAllocMany(n_ancilla);
  QVec qv = qv_w + qv_a;
  // 2. normalize coeffs & prepare ancilla state (PREP)
  float lmbd = 0.0;
  for (auto coeff : coeffs) lmbd += abs(coeff);
  // special case: only one unitary
  if (n_unitary == 1) {
    return block_encoding_res(A / lmbd, lmbd);
  }
  vector<double> probs;
  for (auto coeff : coeffs) probs.push_back(abs(coeff) / lmbd);
  vector<double> amplitude;
  for (auto prob : probs) amplitude.push_back(sqrt(prob));
  QCircuit PREP = amplitude_encode(qv_a, amplitude, false);
  // 3. make selection on unitaries (SEL)
  QCircuit SEL;
  for (int i = 0; i < terms.size(); i++) {
    // condition
    int r = i;
    QCircuit cond;
    for (int j = 0; j < n_ancilla; j++) {
      if (r % 2 == 0) cond << X(qv_a[j]);
      r /= 2;
    }
    // unitary
    string term = terms[i];
    QCircuit cu;
    int j = 0;    // FIXME: current only support 1-qubit :(
    auto sym = term.size() > 0 ? term[j] : 'I';
    switch (sym) {
      case 'I': cu << I(qv_w[j]).control(qv_a); break;
      case 'X': cu << X(qv_w[j]).control(qv_a); break;
      //case 'Y': cu << Y(qv_w[j]).control(qv_a); break;    // QPanda not support
      case 'Z': cu << Z(qv_w[j]).control(qv_a); break;
    }
    SEL << createEmptyCircuit() << cond << cu << cond.dagger() << BARRIER(qv);
  }
  // 4. build total circuit
  QCircuit qcir = createEmptyCircuit() << PREP << BARRIER(qv) << SEL << PREP.dagger();

  // QCircuit to matrix
  QProg prog = createEmptyQProg() << qcir;
  QStat mat_flat = getCircuitMatrix(prog, true);
  qvm.finalize();
  int N_ex = pow(2, n_qubit_ex);
  MatrixXcd U_A = Map<MatrixXcd>(mat_flat.data(), N_ex, N_ex);

  return block_encoding_res(U_A, 1 / lmbd, qcir);
}


// Block-Encoding via ARCSIN method from arXiv:2402.17529
// Accepting d-sparse square matrix with |a_ij| <= 1
block_encoding_res block_encoding_ARCSIN(MatrixXcd A, float eps=1e-8) {
  if (!is_shape_pow2(A)) throw domain_error("A shape is not power of 2");
  if (!is_elem_norm(A)) throw domain_error("A elememnts must st. |a_ij| <= 1");

  // build circuit
  // 1. decide system size
  int N = A.rows(), M = A.cols();
  size_t n_qubit = ceil(log2(N));
  size_t n_qubit_ex = 1 + n_qubit * 2;
  CPUQVM qvm;
  qvm.setConfigure({n_qubit_ex, n_qubit_ex});
  qvm.init();
  // |c0,...,cn;r0,...,rm;a>, mind the fucking order!! :(
  QVec qv = qvm.qAllocMany(n_qubit_ex);
  QVec qv_col = QVec(qv.begin(), qv.begin() + n_qubit);
  QVec qv_row = QVec(qv.begin() + n_qubit, qv.begin() + 2 * n_qubit);
  QVec qv_mctrl = QVec(qv.begin(), qv.end() - 1);
  Qubit* q_anc = qv[qv.size() - 1];
  // 2. make oracle O_A
  MatrixXd thetas = A.array().abs().asin().matrix();
  MatrixXd phis = A.array().arg().matrix();
  QCircuit O_A;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      float t = 2 * thetas(i, j);
      float p = 2 * phis(i, j);
      if (abs(t) < eps && abs(p) < eps) continue;

      // cond
      QCircuit cond;
      int loc = i;
      for (auto q_r : qv_row) {
        if (loc % 2 == 0) cond << X(q_r);
        loc /= 2;
      }
      loc = j;
      for (auto q_c : qv_col) {
        if (loc % 2 == 0) cond << X(q_c);
        loc /= 2;
      }
      // mctrl-rot
      QCircuit mcrot_circ;
      if (abs(t) > eps) mcrot_circ << RY(q_anc, t).control(qv_mctrl);
      if (abs(p) > eps) mcrot_circ << RZ(q_anc, p).control(qv_mctrl);
      // combine one layer
      O_A << cond << mcrot_circ << cond << BARRIER(qv);
    }
  }
  // 3. make H, SWAP and flip-X
  QCircuit H_circ;
  for (auto q : qv_row) H_circ << H(q);
  QCircuit SWAP_circ;
  for (int k = 0; k < qv_row.size(); k++) SWAP_circ << SWAP(qv_row[k], qv_col[k]);
  SWAP_circ << X(q_anc);
  // 4. build total circuit
  QCircuit qcir = createEmptyCircuit() << H_circ << BARRIER(qv) << O_A << BARRIER(qv) << SWAP_circ << BARRIER(qv) << H_circ;

  // QCircuit to matrix: 穷举制备每个基态，逐列投影出线路所对应的矩阵
  // XXX: getCircuitMatrix() does NOT work for this, IDK why :(
  int N_ex = pow(2, n_qubit_ex);
  MatrixXcd U_A(N_ex, N_ex);
  for (int i = 0; i < N_ex; i++) {
    int r = i;
    QCircuit cond;
    for (int j = 0; j < n_qubit_ex; j++) {
      if (r % 2 == 1) cond << X(qv[j]);
      r /= 2;
    }
    QProg qprog = createEmptyQProg() << cond << qcir;
    qvm.directlyRun(qprog);
    QStat qs = qvm.getQState();
    for (int j = 0; j < N_ex; j++) 
      U_A(j, i) = qs[j];
  }
  qvm.finalize();

  return block_encoding_res(U_A, 1 / float(N), qcir);
}


// Block-Encoding via FABLE method from arXiv:2205.00081
// Accepting d-sparse square matrix with |a_ij| <= 1
// @impl translated from https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/templates/subroutines/fable.py
VectorXd block_encoding_FABLE_compute_theta(VectorXd alpha) {
  int N = alpha.size();
  MatrixXd M_trans = MatrixXd::Zero(N, N);
  for (int i = 0; i < M_trans.rows(); i++)
    for (int j = 0; j < M_trans.cols(); j++) {
      int row = j, col = i;     // XXX: the order seems transposed, IDK why :(
      int b_and_g = row & ((col >> 1) ^ col);
      int sum_of_ones = 0;
      while (b_and_g > 0) {
        if (b_and_g % 2 == 1)
          sum_of_ones += 1;
        b_and_g /= 2;
      }
      M_trans(i, j) = sum_of_ones % 2 ? -1 : 1;
    }
  return M_trans * alpha / N;
}

void block_encoding_FABLE_gray_code_recursive(vector<string> &g, int rank) {
  if (rank <= 0) return;
  size_t k = g.size();
  for (int i = k - 1;i >= 0; i--)
    g.push_back("1" + g[i]);
  for (int i = k - 1;i >= 0; i--)
    g[i] = "0" + g[i];
  block_encoding_FABLE_gray_code_recursive(g, rank - 1);
}

vector<int> block_encoding_FABLE_gray_code(int rank) {
  vector<string> g = {"0", "1"};
  block_encoding_FABLE_gray_code_recursive(g, rank - 1);
  vector<int> code;
  for (auto c : g) code.push_back(stoi(c.c_str(), nullptr, 2));
  return code;
}

block_encoding_res block_encoding_FABLE(MatrixXcd A, float eps=1e-8) {
  if (!is_shape_pow2(A)) throw domain_error("A shape is not power of 2");
  if (!is_real(A)) throw domain_error("A is not real");
  if (!is_elem_norm(A)) throw domain_error("A elememnts must st. |a_ij| <= 1");

  // build circuit
  // 1. decide system size
  int N = A.rows();
  size_t n_qubit = ceil(log2(N));
  size_t n_qubit_ex = 1 + n_qubit * 2;
  CPUQVM qvm;
  qvm.setConfigure({n_qubit_ex, n_qubit_ex});
  qvm.init();
  // |a,i,j(working)>, mind the fucking order!! :(
  QVec qv = qvm.qAllocMany(n_qubit_ex);
  vector<int> wires, wires_i, wires_j;
  for (int i = n_qubit_ex - 1; i >= 0; i--) wires.push_back(i);
  int n_wire = wires.size();
  int anc = wires[0];
  for (int k = n_wire / 2; k >= 1; k--) wires_i.push_back(wires[k]);
  for (int k = n_wire - 1; k >= n_wire / 2 + 1; k--) wires_j.push_back(wires[k]);
  vector<int> code = block_encoding_FABLE_gray_code(2 * n_qubit);
  int n_sel = code.size();
  vector<int> ctrl_wires;
  for (int i = 0; i < n_sel; i++) {
    int idx = log2(code[i] ^ code[(i + 1) % n_sel]);
    ctrl_wires.push_back(idx);
  }
  map<int, int> wire_map; {
    int idx = 0;  // make it tmp var
    for (auto k : wires_j) { wire_map[idx] = k; idx++; }
    for (auto k : wires_i) { wire_map[idx] = k; idx++; }
  }
  // 2. make oracle O_A
  Matrix<double, Dynamic, Dynamic, RowMajor> A_rm(A.real());
  Map<RowVectorXd> A_flat(A_rm.data(), A_rm.size());
  VectorXd alphas = A_flat.real().array().acos();
  VectorXd thetas = block_encoding_FABLE_compute_theta(alphas);
  QCircuit O_A;
  set<int> nots;
  for (int k = 0; k < thetas.size(); k++) {
    float theta = thetas(k);
    int ctrl_idx = ctrl_wires[k];
    if (abs(2 * theta) > eps) {
      for (auto ctrl_wire : nots)
        O_A << CNOT(qv[ctrl_wire], qv[anc]);
      O_A << RY(qv[anc], 2 * theta);
      nots.clear();
    }
    if (nots.find(wire_map[ctrl_idx]) != nots.end())
      nots.erase(wire_map[ctrl_idx]);
    else
      nots.insert(wire_map[ctrl_idx]);
  }
  for (auto ctrl_wire : nots)
    O_A << CNOT(qv[ctrl_wire], qv[anc]);
  // 3. make H and SWAP
  QCircuit H_circ;
  for (auto i : wires_i) H_circ << H(qv[i]);
  QCircuit SWAP_circ;
  for (int k = 0; k < wires_i.size(); k++) SWAP_circ << SWAP(qv[wires_i[k]], qv[wires_j[k]]);
  // 4. build total circuit
  QCircuit qcir = createEmptyCircuit() << H_circ << BARRIER(qv) << O_A << BARRIER(qv) << SWAP_circ << BARRIER(qv) << H_circ;

  // QCircuit to matrix: 穷举制备每个基态，逐列投影出线路所对应的矩阵
  // XXX: getCircuitMatrix() does NOT work for this, IDK why :(
  int N_ex = pow(2, n_qubit_ex);
  MatrixXcd U_A(N_ex, N_ex);
  for (int i = 0; i < N_ex; i++) {
    int r = i;
    QCircuit cond;
    for (int j = 0; j < n_qubit_ex; j++) {
      if (r % 2 == 1) cond << X(qv[j]);
      r /= 2;
    }
    QProg qprog = createEmptyQProg() << cond << qcir;
    qvm.directlyRun(qprog);
    QStat qs = qvm.getQState();
    for (int j = 0; j < N_ex; j++) 
      U_A(j, i) = qs[j];
  }
  qvm.finalize();

  return block_encoding_res(U_A, 1 / float(N), qcir);
}


// ↓↓ keep signature for the contest solution
MatrixXcd block_encoding_method(MatrixXcd H) {
  // 由于测试数据不保证谱范数、系数值域等等条件，我们使用应用范围最广的 QSVT 作为最终提交解答；
  // 即使它不存在理论意义上复杂度低的门线路分解方案 :(
  H = rescale_if_necessary(H);
  return block_encoding(H, BlockEncodingMethod::QSVT);
}
