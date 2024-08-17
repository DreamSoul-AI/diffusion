import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
inits = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alphas = [0.1]*10
z_s = []

#initial distribution
alpha_s = 0.1
plt.figure(figsize=(10, 3))
for i in range(len(inits)):
    x = inits[i]
    if(x-alpha_s <0):
        z = np.random.triangular(0, alpha_s, 2*alpha_s, 10000)
    elif(x+alpha_s > 1):
        z = np.random.triangular(1-2*alpha_s, 1-alpha_s, 1, 10000)
    else:
        z = np.random.triangular(x-alpha_s, x, x+alpha_s, 10000)
    z_s.append(z)
    plt.hist(z, bins=200, alpha=0.2, density=True)
plt.show()

#transition
for alpha_t in alphas:
    plt.figure(figsize=(10, 3))
    z_t = []
    for i in range(len(inits)):
        z = []
        for j in range(len(z_s[i])):
            zs = z_s[i][j]
            if(zs-alpha_t <0):
                zt = np.random.triangular(0, alpha_t, 2*alpha_t, 1)[0]
            elif(zs+alpha_t > 1):
                zt = np.random.triangular(1-2*alpha_t, 1-alpha_t, 1, 1)[0]
            else:
                zt = np.random.triangular(zs-alpha_t, zs, zs+alpha_t, 1)[0]
            z.append(zt)
        z_t.append(z)
        plt.hist(z, bins=200, alpha=0.5, density=True)
    alpha_s = alpha_t
    z_s = z_t
    plt.show()