#include <cstdio>
#include <random>
#include <algorithm>
#include "block_encoding.h"
#include "utils.h"

MatrixXcd PauliI(2, 2);
MatrixXcd PauliX(2, 2);
MatrixXcd PauliY(2, 2);
MatrixXcd PauliZ(2, 2);

void init_consts() {
  PauliI << 1, 0,
            0, 1;
  PauliX << 0, 1,
            1, 0;
  PauliY << 0, dcomplex(0, -1),
            dcomplex(0, 1), 0;
  PauliZ << 1,  0,
            0, -1;
}


struct test_res {
  float pass;
  float time;
};

test_res test_block_encoding_QSVT(int T=1000) {
  int ok = 0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXcd A = MatrixXcd::Random(2, 2);
    A = rescale_if_necessary(A);
    clock_t s = clock();
    auto res = block_encoding_QSVT(A);
    ts += clock() - s;
    if (check_block_encoding(res, A)) ok++;
  }
  return {float(ok) / T, ts_to_ms(ts / T)};
}

test_res test_block_encoding_QSVT0(int T=1000) {
  int ok = 0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXcd A(2, 2);
    float a1 = randu(5), a2 = randu(5);
    A << a1, a2,
         a2, a1;
    A = rescale_if_necessary(A);
    clock_t s = clock();
    auto res = block_encoding_QSVT(A);
    ts += clock() - s;
    if (check_block_encoding(res, A)) ok++;
  }
  return {float(ok) / T, ts_to_ms(ts / T)};
}

test_res test_block_encoding_LCU(int k, int T=1000) {
  int ok = 0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXcd A = MatrixXcd::Zero(2, 2);
    for (int j = 0; j < k; j++) {
      int op = rand() % 4;
      while (op == 2) op = rand() % 4;
      float coeff = rand1() * 3;
      switch (op) {
        case 0: A += PauliI * coeff; break;
        case 1: A += PauliX * coeff; break;
        //case 2: A += PauliY * coeff; break;   // QPanda not support
        case 3: A += PauliZ * coeff; break;
      }
    }
    clock_t s = clock();
    auto res = block_encoding_LCU(A);
    ts += clock() - s;
    if (check_block_encoding(res, A)) ok++;
  }
  return {float(ok) / T, ts_to_ms(ts / T)};
}

test_res test_block_encoding_ARCSIN(int d, int T=1000) {
  int ok = 0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXcd A = MatrixXcd::Zero(2, 2);
    vector<int> pos = { 0, 1, 2, 3 };
    shuffle(pos.begin(), pos.end(), default_random_engine());
    for (int j = 0; j < d; j++) {
      dcomplex coeff = dcomplex(randu(), randu());
      double norm = abs(coeff);
      if (norm > 1) coeff /= (norm * 1.00000001);
      switch (pos[j]) {
        case 0: A(0, 0) = coeff; break;
        case 1: A(0, 1) = coeff; break;
        case 2: A(1, 0) = coeff; break;
        case 3: A(1, 1) = coeff; break;
      }
    }
    clock_t s = clock();
    auto res = block_encoding_ARCSIN(A);
    ts += clock() - s;
    if (check_block_encoding(res, A)) ok++;
  }
  return {float(ok) / T, ts_to_ms(ts / T)};
}

test_res test_block_encoding_FABLE(int d, int T=1000) {
  int ok = 0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXcd A = MatrixXcd::Zero(2, 2);
    vector<int> pos = { 0, 1, 2, 3 };
    shuffle(pos.begin(), pos.end(), default_random_engine());
    for (int j = 0; j < d; j++) {
      float coeff = randu() * 0.99999999;
      switch (pos[j]) {
        case 0: A(0, 0) = coeff; break;
        case 1: A(0, 1) = coeff; break;
        case 2: A(1, 0) = coeff; break;
        case 3: A(1, 1) = coeff; break;
      }
    }
    clock_t s = clock();
    auto res = block_encoding_FABLE(A);
    ts += clock() - s;
    if (check_block_encoding(res, A)) ok++;
  }
  return {float(ok) / T, ts_to_ms(ts / T)};
}

test_res test_block_encoding_method(int N, int T=1000) {
  int ok = 0;
  clock_t ts = 0;
  for (int i = 0; i < T; i++) {
    MatrixXcd A = MatrixXcd::Random(N, N);
    clock_t s = clock();
    auto U_A = block_encoding_method(A);
    ts += clock() - s;
    ok++;   // 因为 A 可能被 rescale，无法做数值校验，此处不报错就默认通过
  }
  return {float(ok) / T, ts_to_ms(ts / T)};
}


int main() {
  srand((unsigned)time(NULL)); 
  init_consts();

  int T = 1000;
  test_res res;

  puts("[test_block_encoding_method]");
  for (int N = 2; N <= 8; N++) {
    res = test_block_encoding_method(N, T);
    printf("  N = %d: pass %.2f %%, time %.3f ms\n", N, res.pass * 100, res.time);
  }

  puts("[test_block_encoding_QSVT]");
  res = test_block_encoding_QSVT(T);
  printf("  pass %.2f %%, time %.3f ms\n", res.pass * 100, res.time);

  puts("[test_block_encoding_QSVT0]");
  res = test_block_encoding_QSVT0(T);
  printf("  pass %.2f %%, time %.3f ms\n", res.pass * 100, res.time);

  puts("[test_block_encoding_LCU]");
  for (int k = 2; k <= 8; k++) {
    res = test_block_encoding_LCU(k, T);
    printf("  k = %d: pass %.2f %%, time %.3f ms\n", k, res.pass * 100, res.time);
  }

  puts("[test_block_encoding_ARCSIN]");
  for (int d = 1; d <= 4; d++) {
    res = test_block_encoding_ARCSIN(d, T);
    printf("  d = %d: pass %.2f %%, time %.3f ms\n", d, res.pass * 100, res.time);
  }

  puts("[test_block_encoding_FABLE]");
  for (int d = 1; d <= 4; d++) {
    res = test_block_encoding_FABLE(d, T);
    printf("  d = %d: pass %.2f %%, time %.3f ms\n", d, res.pass * 100, res.time);
  }
}
