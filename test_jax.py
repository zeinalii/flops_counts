from jax import random
import time
import numpy as np

N = 8192

key = random.PRNGKey(0)

# N^2
A = random.normal(key, shape=(N,N), dtype=np.float32)
# N^2
B = random.normal(key, shape=(N,N), dtype=np.float32)
print('\nGPU/TPU testing: ')
SUM = 0
iterations = 100
for i in range(iterations):
    flop = 2*N*N*N
    start = time.monotonic()
    # N^2
    C = A @ B
    end = time.monotonic()
    flops = flop / (end - start)
    SUM += flops / 1e12
print(f"average: {SUM/iterations:,.2f} TFLOPS")