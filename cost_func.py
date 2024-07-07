import numpy as np
import matplotlib.pyplot as plt

# define cost function
def cost(N, M):
    return N * (np.log2(N) + 1) / np.maximum(N - M + 1, 0)


M = np.arange(1, 1000, dtype=int)
N = 2 ** np.arange(4, 14, dtype=int)
nn, mm = np.meshgrid(N, M)
yy = cost(nn, mm).astype(int)
optimal_N = N[np.argmin(yy, axis=1)]

# visualize result
plt.scatter(M, optimal_N)
plt.semilogy(base=2)
plt.yticks(np.unique(optimal_N))
plt.xlabel("Filter length (M)")
plt.ylabel("Optimal FFT size (N)")
plt.show()


# save pairs
last_n = optimal_N[0]
pairs = []
for _m, _n in zip(M, optimal_N):
    if _n == last_n:
        continue
    pairs.append([_m, last_n])
    last_n = _n
pairs = np.array(pairs)

np.savetxt("optimal_fft_size.csv", pairs, fmt="%d", delimiter=",")
