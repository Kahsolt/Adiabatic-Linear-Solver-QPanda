#include "utils.h"

bool is_square(MatrixXcd &A) {
  return A.rows() == A.cols();
}

bool is_posdef(MatrixXcd &A) {
  return A.eigenvalues().cwiseAbs().minCoeff() > 0;
}

bool is_hermitian(MatrixXcd &A) {
  return (A - A.adjoint()).cwiseAbs().maxCoeff() < 1e-8;
}

bool is_unitary(MatrixXcd &A) {
  return (MatrixXcd::Identity(A.rows(), A.cols()) - A.adjoint() * A).cwiseAbs().maxCoeff() < 1e-8;
}


MatrixXcd sqrt(MatrixXcd A) {
  if (is_square(A)) {
    ComplexEigenSolver<MatrixXcd> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) abort();
    auto V = solver.eigenvectors();
    auto D = solver.eigenvalues();
    return V * D.cwiseSqrt().asDiagonal() * V.inverse();
  } else {
    JacobiSVD<MatrixXcd> svd;
    svd.compute(A, ComputeThinU | ComputeThinV);
    auto U = svd.matrixU();
    auto S = svd.singularValues();
    auto V = svd.matrixV();
    return U * S.cwiseSqrt().asDiagonal() * V;
  }
}

float spectral_norm(MatrixXcd &A) {
  if (is_square(A)) {
    return A.eigenvalues().cwiseAbs().maxCoeff();
  } else {
    JacobiSVD<MatrixXcd> svd;
    svd.compute(A, ComputeThinU | ComputeThinV);
    return svd.singularValues().cwiseAbs().maxCoeff();
  }
}

float condition_number(MatrixXcd &A) {
  if (is_square(A)) {
    auto abs_evs = A.eigenvalues().cwiseAbs();
    return abs_evs.maxCoeff() / abs_evs.minCoeff();
  } else {
    JacobiSVD<MatrixXcd> svd;
    svd.compute(A, ComputeThinU | ComputeThinV);
    auto abs_svs = svd.singularValues().cwiseAbs();
    return abs_svs.maxCoeff() / abs_svs.minCoeff();
  }
}


// ↓↓ test utils
float rand1() {
  return float(rand()) / RAND_MAX;
}

float randu(float v=1.0) {
  float r = rand1();
  r = r * 2 - 1;
  return r * v;
}

float ts_to_ms(clock_t ts) {
  return float(ts) / CLOCKS_PER_SEC * 1000.0;
}

MatrixXcd rescale_if_necessary(MatrixXcd A) {
  float norm = spectral_norm(A);
  if (norm <= 1) return A;

  ComplexEigenSolver<MatrixXcd> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) abort();
  auto V = solver.eigenvectors();
  auto D = solver.eigenvalues();
  return V * (D / norm).asDiagonal() * V.inverse();
}
