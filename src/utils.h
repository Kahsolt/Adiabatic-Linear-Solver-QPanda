#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <ctime>
#include "ThirdParty/Eigen/Eigen"
using namespace Eigen;
using namespace std;

bool is_square(MatrixXcd &A);
bool is_posdef(MatrixXcd &A);
bool is_hermitian(MatrixXcd &A);
bool is_unitary(MatrixXcd &A);

MatrixXcd sqrt(MatrixXcd A);
float spectral_norm(MatrixXcd &A);
float condition_number(MatrixXcd &A);

// ↓↓ test utils
float rand1();
float randu(float v=1.0);
float ts_to_ms(clock_t ts);
MatrixXcd rescale_if_necessary(MatrixXcd A);

template <typename T>
void print_vector(vector<T> v, string name) {
  cout << name << ' ';
  for (auto it = v.begin(); it != v.end(); ++it)
    cout << *it << ' ';
  cout << endl;
}

#endif
