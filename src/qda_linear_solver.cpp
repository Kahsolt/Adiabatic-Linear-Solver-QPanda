#include <EigenUnsupported/Eigen/KroneckerProduct>
#include <EigenUnsupported/Eigen/MatrixFunctions>
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "block_encoding.h"
#include "qda_linear_solver.h"
#include "utils.h"

static VectorXcd v0(2);
static MatrixXcd PauliX(2, 2);
static MatrixXcd PauliY(2, 2);
static MatrixXcd PauliP(2, 2);
static MatrixXcd PauliM(2, 2);
static bool is_init_consts = false;

void init_consts() {
  if (is_init_consts) return;
  v0 << 1, 0;
  PauliX << 0, 1,
            1, 0;
  PauliY << 0, dcomplex(0, -1),
            dcomplex(0, 1), 0;
  PauliP = (PauliX + dcomplex(0, 1) * PauliY) / 2;
  PauliM = (PauliX - dcomplex(0, 1) * PauliY) / 2;
  is_init_consts = true;
}

inline MatrixXcd exp_iHt_approx(MatrixXcd H, float t=1.0) {
  // exp(-iHt) ~= I - iH
  int N = H.rows();
  return MatrixXcd::Identity(N, N) - dcomplex(0, 1) * H * t;
}


// The ideal yet naive QDA implementation
VectorXcd linear_solver_ideal(MatrixXcd A, VectorXcd b) {
  init_consts();

  // time-dependent hamiltonian
  int N = A.rows();
  MatrixXcd Qb = MatrixXcd::Identity(N, N) - b * b.adjoint();
  MatrixXcd H0 = kroneckerProduct(PauliX, Qb);
  MatrixXcd H1 = kroneckerProduct(PauliP, A * Qb) + kroneckerProduct(PauliM, Qb * A);
  auto H_s = [&](float s) -> MatrixXcd { return (1 - s) * H0 + s * H1; };
  VectorXd init_state = kroneckerProduct(v0, b).real();  // |0,b>

  // gated quantum computing
  const int S = 200;    // premise 1: adequate stage count
  const int T = 10;     // premise 2: adequate physical time during each stage
  const size_t n_qubit = ceil(log2(H0.rows()));
  CPUQVM qvm;
  qvm.setConfigure({n_qubit, n_qubit});
  qvm.init();
  QVec qv = qvm.qAllocMany(n_qubit);
  QCircuit qcir;
  // init state
  vector<double> amplitude(init_state.data(), init_state.data() + init_state.size());
  qcir << amplitude_encode(qv, amplitude);  // premise 3: correct initial state preparation
  // adiabetic evolution
  for (int s = 0; s < S; s++) {
    MatrixXcd H = H_s(float(s) / S);  // premise 4: ideal time-independent hamiltonion schedule
    MatrixXcd iHt = dcomplex(0, -1) * H * T;
    QMatrixXcd U_iHt = iHt.exp();     // premise 5: ideal time evolution operator
    QCircuit qc_TE = matrix_decompose_qr(qv, U_iHt, false);  // NOTE: must keep false here
    qcir << qc_TE << BARRIER(qv);
  }
  // final state
  QProg qprog = createEmptyQProg() << qcir;
  qvm.directlyRun(qprog);
  QStat qs = qvm.getQState();

  // result
  qs = QStat(qs.begin(), qs.begin() + 2);  // project only the first qubit
  VectorXcd state = Map<VectorXcd>(qs.data(), qs.size());
  return state / state.norm();  // re-norm
}


// The basic implementation strictly follows all contest-specified restrictions (?
VectorXcd linear_solver_contest(MatrixXcd A, VectorXcd b) {
  init_consts();

  // time-dependent hamiltonian
  int N = A.rows();
  MatrixXcd Qb = MatrixXcd::Identity(N, N) - b * b.adjoint();
  MatrixXcd H0 = kroneckerProduct(PauliX, Qb);
  MatrixXcd H1 = kroneckerProduct(PauliP, A * Qb) + kroneckerProduct(PauliM, Qb * A);
  VectorXcd init_state = kroneckerProduct(v0, b);
  /* naive AQC
  auto H_s = [&](float s) -> MatrixXcd { return (1 - s) * H0 + s * H1; };
  */
  /* AQC(P) */
  float k = 5.82842712474619;   // condition_number of the original A
  float p = 2.0;
  auto f_ = [&](float s) -> float {
    float t = 1 + s * (pow(k, p - 1) - 1);
    return k / (k - 1) * (1 - pow(t, 1 / (1 - p)));
  };
  auto H_s = [&](float s) -> MatrixXcd {
    float f_s = f_(s);
    return (1 - f_s) * H0 + f_s * H1;
  };
  
  // gated quantum computing
  const int S = 200;    // restrict 1: fixed as contest required
  const int T = 10;     // restrict 2: fixed as contest required (?)
  const size_t n_qubit = ceil(log2(H0.rows()));   // since we're going to block-encode the hamiltonion H(s), not the matrix A in equation
  const size_t n_ancilla = 1;   // NOTE: modify this according to your block_encode method :)
  const size_t n_qubit_ex = n_ancilla + n_qubit;
  const float lmbd = pow(2, N);
  CPUQVM qvm;
  qvm.setConfigure({n_qubit_ex, n_qubit_ex});
  qvm.init();
  QVec qv = qvm.qAllocMany(n_qubit_ex);
  QCircuit qcir;
  // init state
  VectorXcd anc0 = v0;
  for (int i = 1; i < n_ancilla; i++) anc0 = kroneckerProduct(anc0, v0).eval();   // NOTE: avoid aliasing effect
  VectorXd init_state_ex = kroneckerProduct(anc0, init_state).real();
  vector<double> amplitude(init_state_ex.data(), init_state_ex.data() + init_state_ex.size());
  qcir << amplitude_encode(qv, amplitude);   // |anc_BE,0,b>
  // adiabetic evolution
  for (int s = 0; s < S; s++) {
    MatrixXcd H = H_s((float)s / S);
    MatrixXcd iHt = exp_iHt_approx(H, T);   // restrict 3: approx as contest required
    iHt = normalize_QSVT(iHt);
    QMatrixXcd U_iHt = block_encoding_QSVT(iHt).unitary;  // restrict 4: block_encode as contest required, and as consequence of approx
    QCircuit qc_TE = matrix_decompose_qr(qv, U_iHt, false);  // NOTE: must keep false here
    qcir << qc_TE << BARRIER(qv);
  }
  // final state
  QProg qprog = createEmptyQProg() << qcir;
  qvm.directlyRun(qprog);
  QStat qs = qvm.getQState();

  // result
  qs = QStat(qs.begin(), qs.begin() + 2);  // project only the first qubit
  VectorXcd state = Map<VectorXcd>(qs.data(), qs.size());
  return state / state.norm();  // re-norm
}


// ↓↓ keep signature for the contest solution
#define DEBUG
#ifdef DEBUG
#include <iostream>
using namespace std;
#endif

qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b) {
  // classical solution |x_r>
  VectorXcd s_r = A.colPivHouseholderQr().solve(b);
  VectorXcd x_r = s_r / s_r.norm();
  // quantum solution |x>
  auto b_norm = b.norm();
  VectorXcd x = linear_solver_contest(A / b_norm, b / b_norm);
  // fidelity <x_r|x>
  dcomplex fidelity = x_r.adjoint().dot(x);

#ifdef DEBUG
  cout << "Classical solution: " << x_r.real().transpose() << endl;
  cout << "Quantum solution: " << x.real().transpose() << endl;
  cout << "Fidelity: " << abs(fidelity) << endl;
#endif

  // result pack
  return { x, fidelity };
}
