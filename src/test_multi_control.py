from pyqpanda import CPUQVM, X, RY, BARRIER, QCircuit, QProg

# 单元测试 multi-control-qubits 的有效性

t = 0.1

qvm = CPUQVM()
qvm.init_qvm()
qv = qvm.qAlloc_many(3)
q_anc = qv[0]
qv_idx = qv[1:]
qv_mctrl = qv[1:]

qc = QCircuit()
cond = QCircuit()
qc << cond << RY(q_anc, 1*t).control(qv_mctrl) << cond
qc << BARRIER(qv)
cond = QCircuit() << X(qv_idx[0])
qc << cond << RY(q_anc, 2*t).control(qv_mctrl) << cond
qc << BARRIER(qv)
cond = QCircuit() << X(qv_idx[1])
qc << cond << RY(q_anc, 3*t).control(qv_mctrl) << cond
qc << BARRIER(qv)
cond = QCircuit() << X(qv_idx[0]) << X(qv_idx[1])
qc << cond << RY(q_anc, 4*t).control(qv_mctrl) << cond
qc << BARRIER(qv)

for init_qs in [
  QCircuit() << X(qv_idx[0]) << X(qv_idx[1]),   # |11> will trig CNOT
  QCircuit() << X(qv_idx[1]),                   # |10> will trig I@X CNOT
  QCircuit() << X(qv_idx[0]),                   # |01> will trig X@I CNOT
  QCircuit(),                                   # |00> will trig X@X CNOT
]:
  prog = QProg() << init_qs << BARRIER(qv) << qc
  res = qvm.prob_run_dict(prog, q_anc)
  print(res)
