# API doc


### block_encoding

⚪ `MatrixXcd block_encoding_method(MatrixXcd H)`


### qda_linear_solver

⚪ `VectorXcd linear_solver_method(MatrixXcd A, VectorXcd b)`

⚪ `struct qdals_res { VectorXcd state; complex<double> fidelity; }`

⚪ `qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b);`
