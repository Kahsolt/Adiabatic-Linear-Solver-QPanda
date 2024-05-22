#include <cstdio>
#include "qda_linear_solver.h"
#include "utils.h"

struct test_res {
  float fidelity;
  float time;
};

test_res test_linear_solver_ideal(int T=1000) {
  float fid = 0.0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXd A = MatrixXd::Random(2, 2);
    VectorXd b = VectorXd::Random(2);
    double b_norm = b.norm();
    A = A / b_norm;
    b = b / b_norm;
    clock_t s = clock();
    VectorXcd x = linear_solver_ideal(A, b);
    ts += clock() - s;
    VectorXcd x_r = A.colPivHouseholderQr().solve(b);
    fid += abs(x_r.adjoint().dot(x));
  }
  return {fid / T, ts_to_ms(ts / T)};
}

test_res test_linear_solver_contest(int T=1000) {
  float fid = 0.0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXd A = MatrixXd::Random(2, 2);
    VectorXd b = VectorXd::Random(2);
    double b_norm = b.norm();
    A = A / b_norm;
    b = b / b_norm;
    clock_t s = clock();
    VectorXcd x = linear_solver_contest(A, b);
    ts += clock() - s;
    VectorXcd x_r = A.colPivHouseholderQr().solve(b);
    fid += abs(x_r.adjoint().dot(x));
  }
  return {fid / T, ts_to_ms(ts / T)};
}


int main() {
  srand((unsigned)time(NULL)); 

  int T = 1000;
  test_res res;

  puts("[test_linear_solver_ideal]");
  res = test_linear_solver_ideal(T);
  printf("  fidelity %.7f, time %.3f ms\n", res.fidelity, res.time);

  puts("[test_linear_solver_contest]");
  res = test_linear_solver_contest(T);
  printf("  fidelity %.7f, time %.3f ms\n", res.fidelity, res.time);
}
