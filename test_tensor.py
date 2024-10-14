import tensorflow as tf
import time
import numpy as np

N = 2**14

A = tf.random.uniform(
    (N, N), minval=0, maxval=1, dtype=tf.float32, seed=0)
B = tf.random.uniform(
    (N, N), minval=0, maxval=1, dtype=tf.float32, seed=0) 

with tf.device('GPU:0'):
    SUM = 0
    iterations = 10
    for i in range(iterations):
        flop = 2*N*N*N
        start = time.monotonic()
        # N^2
        C = tf.linalg.matmul(A, B)
        end = time.monotonic()
        flops = flop / (end - start)
        SUM += flops / 1e15
        print(f"{flops / 1e15:,.2f} PFLOPS")
    print(f"average: {SUM/iterations:,.2f} PFLOPS")