#include <iostream>
#include "qda_linear_solver.h"

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
