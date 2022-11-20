import numpy as np
import pytest
import mugrade
import torch
# import needle as ndl
from needle import backend_ndarray as nd


_DEVICES = [nd.cpu(), pytest.param(nd.cuda(),
    marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU"))]

import time

def test_matmul_repeat():
    M, N, P = 2048, 2048, 2048
    repeat = 100
    time_np, time_torch, time_cuda_naive, time_cuda_shared = 0, 0, 0, 0
    cuda = nd.cuda()
    for i in range(repeat):
        # numpy
        _A = np.random.randn(M, N).astype(np.float32)
        _B = np.random.randn(N, P).astype(np.float32)
        time_start = time.time()
        _C = _A @ _B
        time_end = time.time()
        time_np += (time_end-time_start)
        # cuda_shared
        A1 = nd.array(_A, device=cuda)
        B1 = nd.array(_B, device=cuda)
        C1 = nd.NDArray.make((M, P), device=cuda)
        time_start = time.time()
        cuda.matmul(A1._handle, B1._handle, C1._handle, M, N, P)
        time_end = time.time()
        time_cuda_shared += (time_end-time_start)
        np.testing.assert_allclose(C1.numpy(), _C, rtol=1e-3, atol=1e-3)
        # del A1, B1, C1
        # cuda_naive
        A2 = nd.array(_A, device=cuda)
        B2 = nd.array(_B, device=cuda)
        C2 = nd.NDArray.make((M, P), device=cuda)
        time_start = time.time()
        cuda.matmul_naive(A2._handle, B2._handle, C2._handle, M, N, P)
        time_end = time.time()
        time_cuda_naive += (time_end-time_start)
        np.testing.assert_allclose(C2.numpy(), _C, rtol=1e-3, atol=1e-3)
        # del A2, B2, C2
        # torch
        A3 = torch.tensor(_A, device=torch.device('cuda'))
        B3 = torch.tensor(_B, device=torch.device('cuda'))
        time_start = time.time()
        C3 = A3 @ B3
        time_end = time.time()
        time_torch += (time_end - time_start)
        np.testing.assert_allclose(C3.to(torch.device('cpu')).numpy(), _C, rtol=1e-3, atol=1e-3)
        # del A3, B3, C3
    print("numpy:%f s; torch: %f s; cuda_naive:%f s; cuda_shared:%f s"%(time_np, time_torch, time_cuda_naive,
                                                                        time_cuda_shared))
    np.testing.assert_allclose(C2.numpy(), _C, rtol=1e-5, atol=1e-5)

def test_matmul_compare():
    t = 8
    m, n, p = 512,512,512
    M, N, P = m * t, n * t, p * t
    _A = np.random.randn(M, N).astype(np.float32)
    _B = np.random.randn(N, P).astype(np.float32)
    print("M:%d, N:%d, P:%d"%(M, N, P))
    print("------numpy--------")
    time_start = time.time()
    _C = _A @ _B
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')
    print("------CPU_tiled--------")
    cpu = nd.cpu()
    A1 = nd.array(_A, device=cpu)
    B1 = nd.array(_B, device=cpu)
    C1 = nd.NDArray.make((m, p,t,t), device=cpu)
    time_start = time.time()
    cpu.matmul_tiled(A1._handle, B1._handle, C1._handle, m, n, p)
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')
    del A1
    del B1
    del C1
    print("------GPU_naive--------")
    cuda = nd.cuda()
    A2 = nd.array(_A, device=cuda)
    B2 = nd.array(_B, device=cuda)
    C2 = nd.NDArray.make((M, P), device=cuda)
    time_start = time.time()
    cuda.matmul_naive(A2._handle, B2._handle, C2._handle, M, N, P)
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')
    del A2
    del B2
    del C2
    print("------GPU_tiled--------")
    A3 = nd.array(_A, device=cuda)
    B3 = nd.array(_B, device=cuda)
    C3 = nd.NDArray.make((M, P), device=cuda)
    time_start = time.time()
    cuda.matmul(A3._handle, B3._handle, C3._handle, M, N, P)
    time_end = time.time()
    print('time cost:', (time_end - time_start) * 1000, 'ms')

    # np.testing.assert_allclose(C1.numpy(), _C, rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(C2.numpy(), C3.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(C3.numpy(), _C, rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(C3.numpy(), _C, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")