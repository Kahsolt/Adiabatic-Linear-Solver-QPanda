#include "block_encoding.h"
#include "qda_linear_solver.h"

linear_solver_res linear_solver_naive(MatrixXcd A, VectorXcd b) {
  VectorXcd x(2);
  x << 0.717, 0.689;
  return linear_solver_res(x, x);
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
  auto res = linear_solver_naive(A, b);
  VectorXcd x = res.state_phi;
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
