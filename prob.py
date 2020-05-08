import random
import numpy as np
from math import sqrt, pi, e
from scipy.special import erf

def uniform_rand(zero_prob):
    if random.random() > zero_prob:
        return 0
    return int(random.random() * 16)

def gauss(mean, std, zero_prob):
    f = 1 / 2 * (1 + erf((-1 - mean) / std / sqrt(2)))
    if random.random() < zero_prob - f:
        return 0
    val = np.random.normal(loc=mean, scale=std)
    val = round(val)
    return strap(val, 0, 15)

def strap(x, a, b):
    return min(max(a, x), b)


def to_bin(x):
    s = bin(x)[2:]
    return (4 - len(s)) * '0' + s

count = 0
for i in range(int(1e5)):
    # if uniform_rand(0.5) == 0:
    if gauss(5, 2, 0.05) == 0:
        count += 1
print(count / 1e5)


weights = np.array([gauss(5, 5, 0.05) for _ in range(64)])
bin_weights = [to_bin(x) for x in weights]
weights_bit3 = np.array([int(x[0]) for x in bin_weights])
weights_bit2 = np.array([int(x[-3]) for x in bin_weights])
weights_bit1 = np.array([int(x[-2]) for x in bin_weights])
weights_bit0 = np.array([int(x[-1]) for x in bin_weights])

inputs = np.array([gauss(5, 5, 0.05) for _ in range(64)])
bin_inputs = [to_bin(x) for x in inputs]
inputs_bit3 = np.array([int(x[0]) for x in bin_inputs])
inputs_bit2 = np.array([int(x[-3]) for x in bin_inputs])
inputs_bit1 = np.array([int(x[-2]) for x in bin_inputs])
inputs_bit0 = np.array([int(x[-1]) for x in bin_inputs])

# print(weights)
# print(inputs)

print(bin_weights)
print(bin_inputs)

mbit3 = inputs_bit3 * weights_bit3
mbit2 = inputs_bit2 * weights_bit2
mbit1 = inputs_bit1 * weights_bit1
mbit0 = inputs_bit0 * weights_bit0

# print(mbit3)
# print(mbit2)
# print(mbit1)
# print(mbit0)


sum_mbit3 = sum(mbit3)
sum_mbit2 = sum(mbit2)
sum_mbit1 = sum(mbit1)
sum_mbit0 = sum(mbit0)

print(sum_mbit3)
print(sum_mbit2)
print(sum_mbit1)
print(sum_mbit0)

total_val = sum_mbit3 * 2**3 + sum_mbit2 * 2**2 + sum_mbit1 * 2**1 + sum_mbit0 * 2**1


