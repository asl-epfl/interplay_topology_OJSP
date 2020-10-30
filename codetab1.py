"""
This code can be used to generate simulations similar to Table 1 in the following paper:
Vincenzo Matta, Virginia Bordignon, Augusto Santos, Ali H. Sayed, "Interplay Between Topology and Social Learning Over Weak Graphs", IEEE Open Journal of Signal Processing, 2020.

Please note that the code is not generally perfected for performance, but is rather meant to illustrate certain results from the paper. The code is provided as-is without guarantees.

July 2020 (Author: Virginia Bordignon)
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import random
import os
import pandas as pd
from functions import *
#%%
mpl.style.use('seaborn-deep')
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
#%%
NR = 12
NS = [4,4]
Nid = np.cumsum([0]+NS+[NR])
M = 2
N_T = sum(NS)+NR
N_ITER = 10000
N_PLOT = 300
np.random.seed(0)
random.seed(1)
#%%
################################ Build Network Topology ################################
Gr1 = nx.erdos_renyi_graph(NS[0], 0.7, directed = True)
Gr2 = nx.erdos_renyi_graph(NS[1], 0.7, directed = True)
Gr3 = nx.erdos_renyi_graph(NR, 0.7, directed = True)
while not (nx.is_strongly_connected(Gr1) and nx.is_strongly_connected(Gr2) and nx.is_strongly_connected(Gr3)):
    Gr1 = nx.erdos_renyi_graph(NS[0], 0.7, directed = True)
    Gr2 = nx.erdos_renyi_graph(NS[1], 0.7, directed = True)
    Gr3 = nx.erdos_renyi_graph(NR, 0.7, directed = True)
#%%
G1=nx.adjacency_matrix(Gr1)
G2=nx.adjacency_matrix(Gr2)
G3=nx.adjacency_matrix(Gr3)

A1 = build_adjacency_averaging(NS[0], G1)
A2 = build_adjacency_averaging(NS[1], G2)
A3 = build_adjacency_averaging(NR, G3)
#%%
Tsr = np.zeros((sum(NS), NR))
for i in range(NR):
    Tsr[:,i]=np.random.choice(a=[0, 1.0], size=(sum(NS)), p=[.5, .5])
A_full = np.block([[A1,np.zeros((NS[0],NS[1])),Tsr[:NS[0]]],[np.zeros((NS[1],NS[0])),A2, Tsr[NS[0]:sum(NS)]], [np.zeros((NR,sum(NS))), A3]])
#%%
A_full = A_full/np.sum(A_full, axis = 0)
A_lim = np.linalg.matrix_power(A_full,100)
A_full_dec = np.array([[Decimal(x) for x in y] for y in A_full])
#%%
theta = np.arange(1, M+1)
x = np.linspace(-5, 10, 1000)
dt = (max(x) - min(x))/len(x)
var = 1
#%%
muk = np.random.rand(len(NS) + 1, M)/5 - 1/10
muk = np.zeros((len(NS)+1, M))
#%%
L0 = [gaussian(x, theta[0] + m[0], var) for m in muk]
L1 = [gaussian(x, theta[1] + m[1], var) for m in muk]
#%%
TS1 = 0
TS2 = 1
TS3 = 0
#%%
csi = np.zeros((N_T, N_ITER))
for l in range(NS[0]):
    csi[l] = theta[TS1] + np.sqrt(var)*np.random.randn(N_ITER)
for l in range(NS[0],sum(NS)):
    csi[l] = theta[TS2] + np.sqrt(var)*np.random.randn(N_ITER)
for l in range(sum(NS),N_T):
    csi[l] = theta[TS3] + np.sqrt(var)*np.random.randn(N_ITER)

#%%
FS = [gaussian(x, theta[TS1], var), gaussian(x, theta[TS2], var)]
#%%
################################ Run Social Learning ################################
mu_0 = np.random.rand(N_T, M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
#%%
mu = mu_0.copy()
mu = np.array([[Decimal(x) for x in y] for y in mu])

MU = [mu]
L_i = np.zeros((N_T, M))
PSI_DEC=[]

for i in range(N_ITER):
    for k in range(len(NS) + 1):
        L_i[Nid[k] : Nid[k + 1]] = np.array([gaussian(csi[Nid[k] : Nid[k+1], i], t + muk[k,t-1], var) for t in theta]).T

    L_i_dec = np.array([[Decimal(x) for x in y] for y in L_i])
    psi = bayesian_update(L_i_dec, mu)
    decpsi = np.array([[Decimal(x).ln() for x in y] for y in psi])
    mu = np.exp((A_full_dec.T).dot(decpsi))/np.sum(np.exp((A_full_dec.T).dot(decpsi)),axis =1)[:,None]

    MU.append(mu)
    PSI_DEC.append(decpsi)

#%%
################################ Build Perturbed System ################################
y = build_y(PSI_DEC, M, NS, N_ITER, N_T, NR)
D_ = build_matrix_D(NS, NR, FS, L0, L1, dt)
C = build_matrix_C(D_,PSI_DEC[-1], NS, NR, FS, L0, L1, M)
C_inv = np.linalg.pinv(C)
#%%
N_MC = 1000
np.random.seed(0)
#%%
dicsol_pert={}
y_or = y[1]
C_or = C
C_inv_or = np.linalg.pinv(C_or)
sol_or = C_inv_or.dot(y_or.T)
dicsol_pert[r'Unperturbed Solution']={r'$x_{%d,10}$'%(x+1):sol_or[x] for x in range(2)}

d_vector = np.linspace(0.001,.1,5)
for d in d_vector:
    sol_mc=[]
    for i in range(N_MC):
        y_pert = y[1]
        dD = np.zeros((2, 2))
        np.fill_diagonal(dD, np.abs(np.random.randn(2) * d))
        np.fill_diagonal(np.fliplr(dD), np.random.randn(2) * d)
        D_pert = D_ + dD
        while(np.any(D_pert < 0)):
            np.fill_diagonal(np.fliplr(dD), np.random.randn(2)*d)
            D_pert = D_ + dD

        C_pert = build_matrix_C(D_pert, PSI_DEC[-1][sum(NS)+1], NS, NR, FS, L0, L1, M)
        C_inv_pert = np.linalg.pinv(C_pert)
        sol_pert = C_inv_pert.dot(y_pert.T)
        sol_mc.append(sol_pert)
    dmax = np.sqrt(np.mean(np.abs(np.array(sol_mc) - sol_or)**2,axis=0))
    nmax = np.sqrt(np.mean(np.linalg.norm(np.array(sol_mc) - sol_or, axis=1)**2))/np.linalg.norm(sol_or)
    dicsol_pert[r'$\delta x_{%.4f}$' %d] = {r'$x_{%d,10}$'%(x+1):str((dmax[x] * np.sqrt(2)).round(4)) for x in range(2)}
pd.set_option('display.max_columns', 999)
Tab = pd.DataFrame(dicsol_pert)
print(Tab)
#%%
