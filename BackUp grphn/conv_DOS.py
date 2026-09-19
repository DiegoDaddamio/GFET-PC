from graphen import grphn
import numpy as np
import matplotlib.pyplot as plt

G = grphn()

N_bins = 1000
k_point = np.arange(-30000,0,500)[::-1]*-1
n_max = len(k_point)
alph = 10
E_max=8.5
print(k_point)
stock = []


_, _, DOSp_old = G.DOS_chunked(n_points=k_point[0], n_bins=N_bins,E_max=E_max)

for n, N in enumerate(k_point[1:]):
    print(f"{n+1}/{n_max-1}")

    _, _, DOSp_num = G.DOS_chunked(n_points=N, n_bins=N_bins,E_max=E_max)

    
    er = np.abs(DOSp_num[15:-15]-DOSp_old[15:-15])/DOSp_num[15:-15]
    ers = np.sum(er)/len(er)
    stock.append(ers)

    DOSp_old = DOSp_num

np.savetxt(f'ConvNbin{N_bins}.txt', np.column_stack((k_point[1:], stock)), 
           header=f'k_point\tstock|{N_bins}', 
           fmt='%.6f')

data = np.loadtxt('ConvNbin1000.txt')
print(data)

fig, axes = plt.subplots(1, 1, figsize=(10, 6))
axes.plot(data[:,0], data[:,1], 'b-', linewidth=2)

axes.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()