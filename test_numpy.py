import numpy as np
import time


N = 2**14

np.random.seed(0)

# N^2
A = np.random.rand(N, N).astype(np.float32)
# N^2
B = np.random.rand(N, N).astype(np.float32)
print('\nCPU testing: ')
SUM = 0
iterations = 10
for i in range(iterations):
    flop = 2*N*N*N
    start = time.monotonic()
    # N^2
    C = A @ B
    end = time.monotonic()
    flops = flop / (end - start)
    SUM += flops / 1e12
    print(f"{flops / 1e12:,.2f} GFLOPS")
print(f"average: {SUM/iterations:,.2f} GFLOPS")