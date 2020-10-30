"""
This code can be used to generate simulations similar to Figs. 2, 3 and 4 in the following paper:
Vincenzo Matta, Virginia Bordignon, Augusto Santos, Ali H. Sayed, "Interplay Between Topology and Social Learning Over Weak Graphs", IEEE Open Journal of Signal Processing, 2020.

Please note that the code is not generally perfected for performance, but is rather meant to illustrate certain results from the paper. The code is provided as-is without guarantees.

July 2020 (Author: Virginia Bordignon)
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random
from decimal import *
from functions import *
import sys
#%%
mpl.style.use('seaborn-deep')
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
#%%
CONFIG = sys.argv[1]

#%% Setup 2, 3 and 4 correpond to Fig. 2, 3 and 4 respectively.
if CONFIG == '2':
    NS = [12, 4]
elif CONFIG == '3' or CONFIG == '4':
    NS = [4, 4]
else:
    print('Please choose a valid argument among the following: 2, 3, 4.')
    sys.exit()
#%%
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
#%%
M = 3
NR = 12
N_T = sum(NS)+ NR
N_ITER = 120
N_PLOT = N_ITER
np.random.seed(1)
random.seed(1)
################################ Build Network Topology ################################
#%%
if CONFIG == '2':
    np.random.seed(7)
Gr1 = nx.erdos_renyi_graph(NS[0], 0.7, directed = True)
Gr2 = nx.erdos_renyi_graph(NS[1], 0.7, directed = True)
Gr3 = nx.erdos_renyi_graph(4, 0.7, directed = True)
while not (nx.is_strongly_connected(Gr1) and nx.is_strongly_connected(Gr2) and nx.is_strongly_connected(Gr3)):
    Gr1 = nx.erdos_renyi_graph(NS[0], 0.7, directed = True)
    Gr2 = nx.erdos_renyi_graph(NS[1], 0.7, directed = True)
    Gr3 = nx.erdos_renyi_graph(4, 0.7, directed = True)
Gr3 = nx.erdos_renyi_graph(NR, 0.7, directed = True)
while not nx.is_strongly_connected(Gr3):
    Grr = nx.erdos_renyi_graph(NR, 0.7, directed = True)
#%%
G1 = nx.adjacency_matrix(Gr1)
G2 = nx.adjacency_matrix(Gr2)
G3 = nx.adjacency_matrix(Gr3)

A1 = build_adjacency_averaging(NS[0], G1)
A2 = build_adjacency_averaging(NS[1], G2)
A3 = build_adjacency_averaging(NR, G3)
#%%
if CONFIG == '2' or CONFIG == '4':
    Tsr = np.zeros((sum(NS), NR))
    for i in range(NR):
        Tsr[:,i]=np.random.choice(a=[0, 1.0], size=(sum(NS)), p=[.5, .5])
    A_full = np.block([[A1,np.zeros((NS[0],NS[1])),Tsr[:NS[0]]],[np.zeros((NS[1],NS[0])),A2, Tsr[NS[0]:sum(NS)]], [np.zeros((NR,sum(NS))), A3]])

if CONFIG == '3':
    Tsr1 = np.zeros((NS[0], NR))
    for i in range(NR):
        Tsr1[:,i]=np.random.choice(a=[0, 1.0], size=(NS[0]), p=[.1, .9])
    Tsr2 = np.zeros((NS[1], NR))
    for i in range(NR):
        Tsr2[:,i]=np.random.choice(a=[0, 1.0], size=(NS[1]), p=[.9, .1])
    A_full = np.block([[A1,np.zeros((NS[0],NS[1])),Tsr1],[np.zeros((NS[1],NS[0])),A2, Tsr2], [np.zeros((NR,sum(NS))), A3]])
#%%
A_full = A_full/np.sum(A_full, axis = 0)
G_full = nx.from_numpy_array(A_full, create_using=nx.MultiDiGraph())
#%%
rpos = np.array([[-1,3],[-0.4, 6], [-3.3, 2],[1, 2.5], [2, 3.5], [-0.5, 4.5], [-3, 5], [-2, 4.5], [1, 1.2], [-.7, 1], [-2, 1.5], [1, 5]])
rpos[:,0]=rpos[:,0]+4
rpos[:,1]=rpos[:,1]-8
fixedpos = {i: np.array([0,3])+(5*(np.random.rand(2)-0.5)) for i in np.arange(NS[0])}
fixedpos.update({i: np.array([4,3])+(3*(np.random.rand(2)-0.5)) for i in np.arange(NS[0], sum(NS))})
fixedpos.update({i: np.array([2,-3])+(NR*(np.random.rand(2)-0.5)) for i in np.arange(sum(NS), N_T)})

if CONFIG == '4' or CONFIG == '3':
    fixedpos = {0: [0,3], 1: [0.6, 6], 2: [-2.3, 2], 3: [2, 2.5]}
elif CONFIG == '2':
    fixedpos = {0: [-1,3], 1: [-0.4, 6], 2: [-3.3, 2], 3: [1, 2.5], 4: [2, 3.5], 5: [-0.5, 4.5], 6: [-3, 5], 7: [-2, 4.5], 8: [1, 1.2], 9: [-.7, 1], 10: [-2, 1.5], 11: [1, 5]}

fixedpos.update({NS[0]: [6, 2], NS[0]+1: [7, 6], NS[0]+2: [5, 3.5], NS[0]+3: [8, 3]})
fixedpos.update({j: rpos[j-sum(NS)] for j in range(sum(NS), sum(NS)+NR)})
pos = nx.spring_layout(G_full, fixed = fixedpos.keys(), pos = fixedpos)
#%%
f,ax=plt.subplots(1,1, figsize=(4.5,4.5))
ax.set_xlim(-5.4,11.4)
ax.set_ylim(-8.8,8)
plt.axis('off')
if CONFIG == '2':
    circle1 = mpl.patches.Circle((-.7, 3.5), 3.7, fc='None', ec='k', linewidth=0.8)
else:
    circle1 = mpl.patches.Circle((-.2, 3.5), 3.5, fc='None', ec='k', linewidth=0.8)
circle2 = mpl.patches.Circle((6.7, 3.8), 3, fc='None', ec='k', linewidth=0.8)
circle3 = mpl.patches.Circle((3.3, -4.5), 3.7, fc='None', ec='k', linewidth=0.8)

ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)

nx.draw_networkx_nodes(G_full, pos=pos, node_color= 'C0', vmin=0, vmax= 2, nodelist = range(NS[0]),node_size=300, edgecolors='k', linewidths=.5)
nx.draw_networkx_nodes(G_full, pos=pos, node_color = 'C2', vmin=0, vmax= 2, nodelist = range(NS[0], sum(NS)), node_size=300, edgecolors='k', linewidths=.5)
nx.draw_networkx_nodes(G_full, pos=pos, node_color = '#E2C458', vmin=0, vmax= 2, nodelist = range(sum(NS),N_T), node_size=300, edgecolors='k', linewidths=.5)
nx.draw_networkx_labels(G_full,pos,{i: i+1 for i in range(N_T)},font_size=13, font_color='black', alpha = 1)
nx.draw_networkx_edges(G_full, pos = pos, node_size=300, alpha=1, arrowsize=6, width=.5);
plt.savefig(FIG_PATH + 'fig{}_panel1.pdf'.format(CONFIG), bbox_inches='tight')
#%%
################################ Run Social Learning ################################
theta = np.arange(M) - 1 # space of hypotheses
x = np.linspace(-8, 8, 1000)
dt = (max(x)-min(x))/len(x)
var = 1
#%%
L0 = gaussian(x, theta[0], var)
L1 = gaussian(x, theta[1], var)
L2 = gaussian(x, theta[2], var)
#%%
TS1 = 0
TS2 = 2
TS3 = 1
#%%
csi = np.zeros((N_T, N_ITER))
for l in range(NS[0]):
    csi[l] = theta[TS1]+np.sqrt(var)*np.random.randn(N_ITER)
for l in range(NS[0],sum(NS)):
    csi[l] = theta[TS2]+np.sqrt(var)*np.random.randn(N_ITER)
for l in range(sum(NS),N_T):
    csi[l] = theta[TS3]+np.sqrt(var)*np.random.randn(N_ITER)
#%%
mu_0 = np.random.rand(N_T, M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
#%%
A_full_dec = np.array([[Decimal(x) for x in y] for y in A_full])
#%%
mu = mu_0.copy()
mu = np.array([[Decimal(x) for x in y] for y in mu])
MU = [mu]
L_i=np.zeros((N_T, M))
PSI_DEC=[]
for i in range(N_ITER):
    L_i[:NS[0]] = np.array([gaussian(csi[:NS[0],i], t, var) for t in theta]).T
    L_i[NS[0]:sum(NS)] = np.array([gaussian(csi[NS[0]:sum(NS),i], t, var) for t in theta]).T
    L_i[sum(NS):N_T] = np.array([gaussian(csi[sum(NS):N_T,i], t, var) for t in theta]).T

    L_i_dec = np.array([[Decimal(x) for x in y] for y in L_i])
    psi = bayesian_update(L_i_dec, mu)
    decpsi = np.array([[Decimal(x).ln() for x in y] for y in psi])
    mu = np.exp((A_full_dec.T).dot(decpsi))/np.sum(np.exp((A_full_dec.T).dot(decpsi)),axis =1)[:,None]
    MU.append(mu)
    PSI_DEC.append(decpsi)
#%%
plt.figure(figsize=(5,4.5))
for i in range(sum(NS), sum(NS)+4):
    plt.subplot(2,2,i-sum(NS)+1)
    h = plt.plot([MU[k][i,:] for k in range(N_PLOT)] )
    plt.title('Agent {}'.format(i+1), fontsize=14)
    plt.xlabel('$i$', fontsize=14)
    plt.ylabel(r'$\mu_{{{k},i}}(\theta)$'.format(k=i+1), fontsize=15)
plt.figlegend(h,[r'$\theta=1$',r'$\theta=2$',r'$\theta=3$'], fontsize=14, loc = 1, ncol=3, bbox_to_anchor=(0.45, -.39, 0.5, 0.5))
plt.tight_layout()
plt.subplots_adjust(bottom=.19,wspace=.45, right = .98)
plt.savefig(FIG_PATH+'fig{}_panel2.pdf'.format(CONFIG), bbox_inches='tight')
