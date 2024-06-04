#include <cstdio>
#include "qda_linear_solver.h"
#include "utils.h"


int main() {
  srand((unsigned)time(NULL)); 

  int n_methods = 3;
  int T = 1000;

  vector<clock_t> ts;
  vector<float> fid;
  for (int i = 0; i < n_methods; i++) {
    ts.push_back(0);
    fid.push_back(0.0);
  }

  VectorXcd x_r;
  VectorXcd x;
  clock_t s = -1;
  for (int i = 0; i < T; i++) {
    MatrixXd A = MatrixXd::Random(2, 2);
    VectorXd b = VectorXd::Random(2);
    A *= A.adjoint();  // arbitrary real symmetric matrix
    b /= b.norm();     // arbitrary unit vector

    // classic (ref)
    x_r = A.colPivHouseholderQr().solve(b);

    // ideal
    s = clock();
    x = linear_solver_ideal(A, b, DecompositionMode::QSD);
    ts[0] += clock() - s;
    fid[0] += abs(x_r.adjoint().dot(x));

    // contest
    s = clock();
    x = linear_solver_contest(A, b, DecompositionMode::QSD);
    ts[1] += clock() - s;
    fid[1] += abs(x_r.adjoint().dot(x));

    // ours
    s = clock();
    x = linear_solver_ours(A, b, DecompositionMode::QSD);
    ts[2] += clock() - s;
    fid[2] += abs(x_r.adjoint().dot(x));
  }

  puts("[test_linear_solver_ideal]");
  printf("  fidelity: %.7f, time: %.3f ms\n", fid[0] / T, ts_to_ms(ts[0] / T));
  puts("[test_linear_solver_contest]");
  printf("  fidelity: %.7f, time: %.3f ms\n", fid[1] / T, ts_to_ms(ts[1] / T));
  puts("[test_linear_solver_ours]");
  printf("  fidelity: %.7f, time: %.3f ms\n", fid[2] / T, ts_to_ms(ts[2] / T));
}
