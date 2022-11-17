import numpy as np
import pytest
import mugrade
# import needle as ndl
from needle import backend_ndarray as nd


_DEVICES = [nd.cpu(), pytest.param(nd.cuda(),
    marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU"))]

import time

def test_matmul_compare():
    t = 8
    m, n, p = 256,256,256
    M, N, P = m * t, n * t, p * t
    _A = np.random.randn(M, N)
    _B = np.random.randn(N, P)
    print("M:%d, N:%d, P:%d"%(M, N, P))
    print("------numpy--------")
    time_start = time.time()
    _C = _A @ _B
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')
    print("------CPU_naive--------")
    cpu = nd.cpu()
    A1 = nd.array(_A, device=cpu)
    B1 = nd.array(_B, device=cpu)
    time_start = time.time()
    C1 = A1 @ B1
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')
    print("------CPU_tiled--------")
    A2 = nd.array(_A, device=cpu).reshape((m,n,t,t))
    B2 = nd.array(_B, device=cpu).reshape((n,p, t,t))
    C2 = nd.NDArray.make((m, p, t, t), device=cpu)
    time_start = time.time()
    cpu.matmul_tiled(A2._handle, B2._handle, C2._handle, M, N, P)
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')
    print("------GPU_tiled--------")
    cuda = nd.cuda()
    A3 = nd.array(_A, device=cuda)
    B3 = nd.array(_B, device=cuda)
    time_start = time.time()
    C3 = A1 @ B1
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')

    np.testing.assert_allclose(C1.numpy(), _C, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(C3.numpy(), _C, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")