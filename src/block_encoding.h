#ifndef BLOCK_ENCODING_H
#define BLOCK_ENCODING_H

#include "ThirdParty/Eigen/Eigen"
#include "Core/QuantumCircuit/QCircuit.h"
using namespace Eigen;
using namespace QPanda;

struct block_encoding_res {
  MatrixXcd unitary;
  float subfactor = 1.0;
  QCircuit circuit;
  block_encoding_res() {}
  block_encoding_res(MatrixXcd unitary, float subfactor=1.0): unitary(unitary), subfactor(subfactor) {}
  block_encoding_res(MatrixXcd unitary, float subfactor, QCircuit circuit): unitary(unitary), subfactor(subfactor), circuit(circuit) {}
};

bool check_block_encoding(block_encoding_res &res, MatrixXcd &A, float eps=1e-5);

enum struct BlockEncodingMethod {
  // directly construct
  QSVT   = 10,
  QSVT0  = 11,
  // prepare-select based
  LCU    = 20,
  // query-oracle based
  ARCSIN = 30,
  FABLE  = 31,
};

MatrixXcd block_encoding(MatrixXcd A, BlockEncodingMethod method=BlockEncodingMethod::ARCSIN);

block_encoding_res block_encoding_QSVT(MatrixXcd A);
block_encoding_res block_encoding_QSVT0(MatrixXcd A);
block_encoding_res block_encoding_LCU(MatrixXcd A, float eps=1e-8);
block_encoding_res block_encoding_ARCSIN(MatrixXcd A, float eps=1e-8);
block_encoding_res block_encoding_FABLE(MatrixXcd A, float eps=1e-8);


// ↓↓ keep signature for the contest solution
MatrixXcd block_encoding_method(MatrixXcd A);

#endif
