#ifndef QDA_LINEAR_SOLVER_H
#define QDA_LINEAR_SOLVER_H

#include <complex>
#include "ThirdParty/Eigen/Eigen"
using namespace std;
using namespace Eigen;

VectorXcd linear_solver_ideal(MatrixXcd A, VectorXcd b);
VectorXcd linear_solver_contest(MatrixXcd A, VectorXcd b);


// ↓↓ keep signature for the contest solution
struct qdals_res {
  VectorXcd state;
  complex<double> fidelity;
};

qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b);

#endif
