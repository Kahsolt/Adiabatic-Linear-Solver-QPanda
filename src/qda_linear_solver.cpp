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


// A ideal and simple QDA implementation
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
  const int S = 200;
  const int T = 10;
  const size_t n_qubit = ceil(log2(H0.rows()));
  CPUQVM qvm;
  qvm.setConfigure({n_qubit, n_qubit});
  qvm.init();
  QVec qv = qvm.qAllocMany(n_qubit);
  QCircuit qcir;
  // init state
  vector<double> amplitude(init_state.data(), init_state.data() + init_state.size());
  qcir << amplitude_encode(qv, amplitude);
  // adiabetic evolution
  for (int s = 0; s < S; s++) {
    MatrixXcd H = H_s(float(s) / S);
    MatrixXcd iHt = dcomplex(0, -1) * H * T;
    QMatrixXcd U_iHt = iHt.exp();
    QCircuit qc_TE = matrix_decompose_qr(qv, U_iHt, false);  // NOTE: must keep false here
    qcir << qc_TE << BARRIER(qv);
  }
  // final state
  QProg qprog = createEmptyQProg() << qcir;
  qvm.directlyRun(qprog);
  QStat qs = qvm.getQState();

  // result
  qs = QStat(qs.begin(), qs.begin() + 2);  // project only the firt qubit
  VectorXcd state = Map<VectorXcd>(qs.data(), qs.size());
  return state;
}


// The basic implementation strictly follows all contest-specified restrictions :)
VectorXcd linear_solver_contest(MatrixXcd A, VectorXcd b) {
  init_consts();

  // time-dependent hamiltonian
  int N = A.rows();
  MatrixXcd Qb = MatrixXcd::Identity(N, N) - b * b.adjoint();
  MatrixXcd H0 = kroneckerProduct(PauliX, Qb);
  MatrixXcd H1 = kroneckerProduct(PauliP, A * Qb) + kroneckerProduct(PauliM, Qb * A);
  auto H_s = [&](float s) -> MatrixXcd { return (1 - s) * H0 + s * H1; };
  VectorXcd init_state = kroneckerProduct(v0, b);

  // gated quantum computing
  const int S = 200;
  const int T = 10;
  const size_t n_qubit = ceil(log2(H0.rows()));
  const size_t n_ancilla = 1;   // TODO: modify this
  const size_t n_qubit_ex = n_ancilla + n_qubit;
  CPUQVM qvm;
  qvm.setConfigure({n_qubit_ex, n_qubit_ex});
  qvm.init();
  QVec qv = qvm.qAllocMany(n_qubit_ex);
  QCircuit qcir;
  // init state
  auto anc0 = v0;
  for (int i = 1; i < n_ancilla; i++) anc0 = kroneckerProduct(anc0, v0);
  VectorXd init_state_ex = kroneckerProduct(anc0, init_state).real();
  vector<double> amplitude(init_state_ex.data(), init_state_ex.data() + init_state_ex.size());
  qcir << amplitude_encode(qv, amplitude);   // |anc_BE,0,b>
  // adiabetic evolution
  for (int s = 0; s < S; s++) {
    MatrixXcd H = H_s((float)s / S);
    MatrixXcd iHt = exp_iHt_approx(H, T);
    auto res = block_encoding_QSVT(iHt);
    QMatrixXcd U_iHt = res.unitary;
    QCircuit qc_TE = matrix_decompose_qr(qv, U_iHt, false);  // NOTE: must keep false here
    qcir << qc_TE << BARRIER(qv);
  }
  // final state
  QProg qprog = createEmptyQProg() << qcir;
  qvm.directlyRun(qprog);
  QStat qs = qvm.getQState();

  // result
  qs = QStat(qs.begin(), qs.begin() + 2);  // project only the firt qubit
  VectorXcd state = Map<VectorXcd>(qs.data(), qs.size());
  return state;
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
