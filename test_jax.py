from jax import random
import time
import numpy as np

N = 4096

key = random.PRNGKey(0)

# N^2
A = random.normal(key, shape=(N,N), dtype=np.float32)
# N^2
B = random.normal(key, shape=(N,N), dtype=np.float32)

for i in range(100):
    flop = 2*N*N*N
    start = time.monotonic()
    # N^2
    C = A @ B
    end = time.monotonic()
    flops = flop / (end - start)
    print(f"{flops / 1e9:,.2f} GFLOPS")