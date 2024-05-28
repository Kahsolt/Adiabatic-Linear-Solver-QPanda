#include <EigenUnsupported/Eigen/KroneckerProduct>
#include <EigenUnsupported/Eigen/MatrixFunctions>
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
using namespace std;
using namespace Eigen;
using namespace QPanda;

// 单元测试 matrix_decompose_qr 的精度问题(?)，发现可能是参数 b_positive_seq 语义不一致导致的问题

int main() {
  // test input case
  MatrixXcd A(2, 2);
  A << 2, 1, 
       1, 0;
  VectorXcd b(2);
  b << 3, 1;
  // normalize
  auto b_norm = b.norm();
  A = A / b_norm;
  b = b / b_norm;
  
  // consts
  MatrixXcd PauliX(2, 2);
  MatrixXcd PauliY(2, 2);
  MatrixXcd PauliP(2, 2);
  MatrixXcd PauliM(2, 2);
  PauliX << 0, 1,
            1, 0;
  PauliY << 0, dcomplex(0, -1),
            dcomplex(0, 1), 0;
  PauliP = (PauliX + dcomplex(0, 1) * PauliY) / 2;
  PauliM = (PauliX - dcomplex(0, 1) * PauliY) / 2;

  // time-dependent hamiltonian
  int N = A.rows();
  MatrixXcd Qb = MatrixXcd::Identity(N, N) - b * b.adjoint();
  MatrixXcd H0 = kroneckerProduct(PauliX, Qb);
  MatrixXcd H1 = kroneckerProduct(PauliP, A * Qb) + kroneckerProduct(PauliM, Qb * A);
  auto H_s = [&](float s) -> MatrixXcd { return (1 - s) * H0 + s * H1; };
  auto N_ex = H0.rows();

  // gated quantum computing
  const int S = 200;
  const int T = 10;
  const size_t n_qubit = ceil(log2(N_ex));
  CPUQVM qvm;
  qvm.setConfigure({n_qubit, n_qubit});
  qvm.init();
  QVec qv = qvm.qAllocMany(n_qubit);
  // adiabatic evolution
  vector<double> error_list;
  for (int s = 0; s < S; s++) {
    MatrixXcd H = H_s(s);
    MatrixXcd iHt = dcomplex(0, -1) * H * T;
    MatrixXcd U = iHt.exp();

    // assert U is unitary
    assert(U.isUnitary());
    // MatrixXcd => QStat
    QStat U_qstat = QStat(U.data(), U.data() + U.size());
    // assert U_qstat is unitary
    assert(is_unitary_matrix(U_qstat));
    // matrix => QCircuit
    QCircuit qcir = matrix_decompose_qr(qv, U_qstat, true);   // b_positive_seq=true, 必须与下文设置相反
    // QCircuit => matrix
    QProg prog = createEmptyQProg() << qcir;
    QStat U_qstat_hat = getCircuitMatrix(prog, false);        // b_positive_seq=false, 必须与上文设置相反
    // QStat => MatrixXcd
    MatrixXcd U_hat = Map<MatrixXcd>(U_qstat_hat.data(), N_ex, N_ex);
    // calc L1 error
    double L1_err = (U_hat - U).cwiseAbs().mean();
    cout << "s = " << s << ", err = " << L1_err << endl;

    error_list.push_back(L1_err);
  }
  qvm.finalize();
}
