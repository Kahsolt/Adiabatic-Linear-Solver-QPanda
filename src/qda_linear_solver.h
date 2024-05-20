#ifndef QDA_LINEAR_SOLVER_H
#define QDA_LINEAR_SOLVER_H

#include <complex>
#include "ThirdParty/Eigen/Eigen"
using namespace std;
using namespace Eigen;

struct linear_solver_res {
  VectorXcd state;
  VectorXcd state_phi;
  linear_solver_res() {}
  linear_solver_res(VectorXcd state, VectorXcd state_phi): state(state), state_phi(state_phi) {}
};

linear_solver_res linear_solver_naive(MatrixXcd A, VectorXcd b);


// ↓↓ keep signature for the contest solution
struct qdals_res {
  VectorXcd state;
  complex<double> fidelity;
};

qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b);

#endif
