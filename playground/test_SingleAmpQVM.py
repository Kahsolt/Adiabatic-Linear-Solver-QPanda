#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/06/06 

# 测试单振幅计算是否结果上等价于全振幅计算
# ref: https://pyqpanda-toturial.readthedocs.io/zh/latest/6.%E6%A8%A1%E6%8B%9F%E5%99%A8/index.html#id21
# - 会有部分结果差一个负号的问题，绝对值是匹配的

import numpy as np
from pyqpanda import *

nq = 4

qvm = CPUQVM()
qvm.init_qvm()
sqvm = SingleAmpQVM()
sqvm.init_qvm()

qv = qvm.qAlloc_many(nq)
qc = QCircuit() \
  << RX(qv[0], 0.1) \
  << CNOT(qv[0], qv[1]) \
  << RX(qv[1], 0.2) \
  << CNOT(qv[1], qv[2]) \
  << RZ(qv[2], 0.3) \
  << CNOT(qv[2], qv[3]) \
  << iSWAP(qv[3], qv[0])

qvm.directly_run(qc)
qs = qvm.get_qstate()
print(np.asarray(qs).round(4))

sqs = []
for idx in range(2**nq):
  sqvm.run(qc, qv)    # 跑一次只能测一次
  sqs.append(sqvm.pmeasure_dec_amplitude(str(idx)))
print(np.asarray(sqs).round(4))
