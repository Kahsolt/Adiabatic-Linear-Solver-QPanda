#include <cstdio>
#include "qda_linear_solver.h"
#include "utils.h"

struct test_res {
  float error;
  float time;
};

test_res test_linear_solver_naive(int T=1000) {
  return { 99999, 114514 };
}


int main() {
  srand((unsigned)time(NULL)); 

  int T = 1000;
  test_res res;

  puts("[test_linear_solver_naive]");
  res = test_linear_solver_naive(T);
  printf("  error %.7f, time %.3f ms\n", res.error, res.time);
}
