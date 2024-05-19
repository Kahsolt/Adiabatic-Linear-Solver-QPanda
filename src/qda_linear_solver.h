#pragma once
#include "ThirdParty/Eigen/Eigen"
using namespace Eigen;
using namespace std;

VectorXcd linear_solver_method(MatrixXcd A, VectorXcd b);

typedef struct qdals_res {
  VectorXcd state;
  complex<double> fidelity;
};

qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b);
