# flake8: noqa
import sys
import time

sys.path.append("./research/berry")
import berrylib.fast_inla as fast_inla
import test_berry
import torch


def benchmark_m1_torch():
    for device in ["cpu", "mps"]:
        print("")
        start = time.time()
        N = 5000
        A = torch.rand(N, N, dtype=torch.float).to(device)
        B = torch.rand(N, N, dtype=torch.float).to(device)
        print("create", time.time() - start)

        start = time.time()
        for i in range(100):
            A = torch.mm(A, B) * (2 / N)
        cpu_arr = A.to("cpu")
        print(cpu_arr[0, 0], cpu_arr[-1, -1])
        print("matvec", time.time() - start)
        start = time.time()


N = 10000
it = 4
print("jax")
test_berry.test_fast_inla("jax", N, it)
# print("cpp")
# test_berry.test_fast_inla("cpp", N, it)
# print("numpy")
# test_berry.test_fast_inla("numpy", N, it)
print("pytorch")
test_berry.test_fast_inla("pytorch", N, it)
