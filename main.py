import numpy as np
import time


N = 4096

np.random.seed(0)

# N^2
A = np.random.rand(N, N).astype(np.float32)
# N^2
B = np.random.rand(N, N).astype(np.float32)



for i in range(100):
    flop = 2*N*N*N
    start = time.monotonic()
    # N^2
    C = A @ B
    end = time.monotonic()
    flops = flop / (end - start)
    print(f"{flops / 1e9:,.2f} GFLOPS")