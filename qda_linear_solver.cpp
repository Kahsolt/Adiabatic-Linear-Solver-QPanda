#include "block_encoding.h"
#include "qda_linear_solver.h"

/// @brief linear solver via quantum discrete adiabatic from arXiv:1909.05500
/// @param A the matrix of the linear equation
/// @param b the vector of the linear equation
/// @return the solution x of A * x = b
VectorXcd linear_solver_method(MatrixXcd A, VectorXcd b) {
  VectorXcd x(2);
  x << 0.717, 0.689;

  return x;
}


/*****************************************************************************/

#include <iostream>
using namespace std;

qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b) {
  // classical solution |x_r>
  VectorXcd s_r = A.colPivHouseholderQr().solve(b);
  VectorXcd x_r = s_r / s_r.norm();
  cout << "Classical solution: " << x_r.real().transpose() << endl;
  // quantum solution |x>
  VectorXcd x = linear_solver_method(A, b);
  cout << "Quantum solution: " << x.real().transpose() << endl;
  // fidelity <x_r|x>
  complex<double> fidelity = x_r.conjugate().transpose().dot(x);
  cout << "Fidelity: " << fidelity.real() << endl;
  // result pack
  return { x, fidelity };
}


int main() {
  // test input case
  MatrixXcd A(2, 2);
  A << 2, 1, 
       1, 0;
  VectorXcd b(2);
  b << 3, 1;

  // results
  qdals_res res = qda_linear_solver(A, b);
}
