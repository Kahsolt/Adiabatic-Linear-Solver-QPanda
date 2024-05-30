#ifndef QDA_LINEAR_SOLVER_H
#define QDA_LINEAR_SOLVER_H

#include <complex>
#include "ThirdParty/Eigen/Eigen"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
using namespace std;
using namespace Eigen;
using namespace QPanda;

VectorXcd linear_solver_ideal(MatrixXcd A, VectorXcd b, DecompositionMode decompose_method=DecompositionMode::QR);
VectorXcd linear_solver_contest(MatrixXcd A, VectorXcd b, DecompositionMode decompose_method=DecompositionMode::QR);
VectorXcd linear_solver_ours(MatrixXcd A, VectorXcd b, DecompositionMode decompose_method=DecompositionMode::QR);


// ↓↓ keep signature for the contest solution
struct qdals_res {
  VectorXcd state;
  complex<double> fidelity;
};

qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b);

#endif
