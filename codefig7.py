"""
This code can be used to generate simulations similar to Fig. 7 in the following paper:
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
from decimal import *
from functions import *
#%%
mpl.style.use('seaborn-deep')
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Computer Modern Roman'
#%%
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)
#%%
NR = 12
NS = [4,4,4]
Nid = np.cumsum([0]+NS+[NR])
M = 3
N_T = sum(NS)+NR
N_ITER = 10000
N_PLOT = 300
np.random.seed(10)
random.seed(0)
#%%
################################ Build Network Topology ################################
Gr = [nx.erdos_renyi_graph(n, 0.7) for n in NS + [4]]
while not all([nx.is_connected(g) for g in Gr]):
    Gr = [nx.erdos_renyi_graph(n, 0.7) for n in NS + [4]]

Gr[-1] = nx.erdos_renyi_graph(NR, 0.7)
while not nx.is_connected(Gr[-1]):
    Gr[-1] = nx.erdos_renyi_graph(NR, 0.7)
#%%
G= [nx.adjacency_matrix(gr) for gr in Gr]
A = [build_adjacency_metropolis(g.shape[0], g) for g in G]
#%%
Tsr = np.zeros((sum(NS), NR))
for i in range(NR):
    Tsr[:,i]=np.random.choice(a=[0, 1.0], size=(sum(NS)), p=[.5, .5])
#%%
A_full = sp.linalg.block_diag(*A)
A_full[:sum(NS),sum(NS):]=Tsr
A_full = A_full/np.sum(A_full, axis = 0)
A_full_dec = np.array([[Decimal(x) for x in y] for y in A_full])
A_lim = np.linalg.matrix_power(A_full,100)
G_full = nx.from_numpy_array(A_full, create_using=nx.MultiDiGraph())
#%%
rpos = np.array([[-1,3],[-0.4, 6], [-3.3, 2],[1, 2.5], [2, 3.5], [-0.5, 4.5], [-3, 5], [-2, 4.5], [1, 1.2], [-.7, 1], [-2, 1.5], [1, 5]])
rpos[:,0]=rpos[:,0]+5
rpos[:,1]=rpos[:,1]-8
fixedpos = {i: np.array([0,3])+(5*(np.random.rand(2)-0.5)) for i in np.arange(NS[0])}
fixedpos.update({i: np.array([4,3])+(3*(np.random.rand(2)-0.5)) for i in np.arange(NS[0], NS[0]+NS[1])})
fixedpos.update({i: np.array([2,-3])+(NR*(np.random.rand(2)-0.5)) for i in np.arange(sum(NS), N_T)})
fixedpos = {0: [0,3], 1: [0.6, 6], 2: [-2.3, 2], 3: [2, 2.5]}
fixedpos.update({NS[0]: [6, 2], NS[0]+1: [7, 6], NS[0]+2: [5, 3.5], NS[0]+3: [8, 3]})
fixedpos.update({NS[1]+NS[0]: [13, 2], NS[1]+NS[0]+1: [12, 5],NS[1]+ NS[0]+2: [14, 4], NS[1]+NS[0]+3: [11, 2]})
fixedpos.update({j: rpos[j-12] for j in range(sum(NS), sum(NS)+NR)})
pos = nx.spring_layout(G_full, fixed = fixedpos.keys(), pos = fixedpos)
#%%
f,ax=plt.subplots(1,1, figsize=(5.85,4.5))
ax.set_xlim(-5.4,16.2)
ax.set_ylim(-8.8,8)
plt.axis('off')
nx.draw_networkx_nodes(G_full, pos=pos, node_color= 'C0', vmin=0, vmax= 2, nodelist = range(NS[0]),node_size=350, edgecolors='k', linewidths=.5)
nx.draw_networkx_nodes(G_full, pos=pos, node_color = 'C1', vmin=0, vmax= 2, nodelist = range(NS[0], NS[0]+NS[1]), node_size=350, edgecolors='k', linewidths=.5)
nx.draw_networkx_nodes(G_full, pos=pos, node_color = 'C2', vmin=0, vmax= 2, nodelist = range(NS[0]+NS[1],sum(NS)), node_size=350, edgecolors='k', linewidths=.5)
nx.draw_networkx_nodes(G_full, pos=pos, node_color = '#E2C458', vmin=0, vmax= 2, nodelist = range(sum(NS),N_T), node_size=350, edgecolors='k', linewidths=.5)

circle1 = mpl.patches.Circle((-.2, 3.5), 3.3, fc='None', ec='k', linewidth=0.8)
circle2 = mpl.patches.Circle((6.5, 3.9), 2.8, fc='None', ec='k', linewidth=0.8)
circle3 = mpl.patches.Circle((4.3, -4.5), 3.7, fc='None', ec='k', linewidth=0.8)
circle4 = mpl.patches.Circle((12.4, 3.2), 2.6, fc='None', ec='k', linewidth=0.8)

ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)

nx.draw_networkx_labels(G_full,pos,{i: i+1 for i in range(N_T)},font_size=14, font_color='black', alpha = 1)
nx.draw_networkx_edges(G_full, pos = pos, node_size=350, alpha=1, arrowsize=6, width=0.5);
plt.savefig(FIG_PATH + 'fig7_panel1.pdf', bbox_inches='tight')
#%%
np.random.seed(6)
theta = np.arange(1, M+1)
x = np.linspace(-5, 10, 1000)
dt = (max(x)-min(x))/len(x)
var = 1
#%%
E = np.ones((len(NS)+1)*M)*0.01+np.diag(np.ones((len(NS)+1)*M)*0.01)
muk = np.random.multivariate_normal(np.zeros((len(NS)+1)*M),E)
muk = muk.reshape(len(NS)+1,M)
#%%
L0 = [gaussian(x, theta[0]+m[0], var) for m in muk]
L1 = [gaussian(x, theta[1]+m[1], var) for m in muk]
L2 = [gaussian(x, theta[2]+m[2], var) for m in muk]
#%%
TS = np.array([0,0,0,0])
#%%
csi = np.zeros((N_T, N_ITER))
for l in range(sum(NS)):
    csi[l] = theta[TS[l//4]]+np.sqrt(var)*np.random.randn(N_ITER)
for l in range(sum(NS),N_T):
    csi[l] = theta[TS[3]]+np.sqrt(var)*np.random.randn(N_ITER)
#%%
FS = [gaussian(x, theta[ts], var) for ts in TS[:-1]]
#%%
################################ Run Social Learning ################################
mu_0 = np.random.rand(N_T, M)
mu_0 = mu_0/np.sum(mu_0, axis = 1)[:, None]
#%%
mu = mu_0.copy()
mu = np.array([[Decimal(x) for x in y] for y in mu])
MU = [mu]
L_i=np.zeros((N_T, M))
PSI_DEC=[]
for i in range(N_ITER):
    for k in range(len(NS)+1):
        L_i[Nid[k]:Nid[k+1]] = np.array([gaussian(csi[Nid[k]:Nid[k+1],i], t+muk[k,t-1], var) for t in theta]).T

    L_i_dec = np.array([[Decimal(x) for x in y] for y in L_i])
    psi = bayesian_update(L_i_dec, mu)
    decpsi = np.array([[Decimal(x).ln() for x in y] for y in psi])
    mu = np.exp((A_full_dec.T).dot(decpsi))/np.sum(np.exp((A_full_dec.T).dot(decpsi)),axis =1)[:,None]

    MU.append(mu)
    PSI_DEC.append(decpsi)
#%%
################################ Estimate Weights ################################
psi_dec = [np.array([PSI_DEC[i][x,[k for k in range(M) if k !=np.argmax(PSI_DEC[i][sum(NS):],axis=1)[(x-sum(NS))]]]/(i+1) for i in range(len(PSI_DEC))]) for x in range(sum(NS), N_T)]
ones = np.array([Decimal(1) for i in range(N_ITER)])[:,None]
psi_dec_const= [np.hstack([psi_dec[i], ones]) for i in range(len(psi_dec))]
#%%
DD=[]
for i in range(N_ITER):
    TR_=np.argmax(PSI_DEC[i][sum(NS):], axis=1)
    D_=[]
    for k in range(NR):
        LR =[L0, L1, L2][TR_[k]]
        D_.append(np.array([[np.sum(Y*np.log(Y/LR[iY])*dt)-np.sum(Y*np.log(Y/X)*dt) for X in [L0[iY], L1[iY], L2[iY]]] for iY, Y in enumerate(FS)] ).T)
    DD.append(D_)
#%%
for i in range(N_ITER):
    for j in range(NR):
        DD[i][j]=DD[i][j][np.where(DD[i][j].any(axis=1))[0]]
        DD[i][j]=np.vstack([DD[i][j], np.ones(3)])
#%%
DD_invdec=[]
for i in range(N_ITER):
    D_inv = [np.linalg.pinv(d) for d in DD[i]]
    D_invdec = [np.array([[Decimal(x) for x in y] for y in dinv]) for dinv in D_inv]
    DD_invdec.append(D_invdec)
#%%
sol = [np.array([DD_invdec[j][x].dot(psi_dec_const[x][j]) for j in range(N_ITER)]) for x in range(NR)]
#%%
plt.figure(figsize=(5,4.5))
for i in range(4):
    plt.subplot(2,2,i+1)

    plt.plot(np.ones(len(sol[i]))*A_lim[:NS[0],i+sum(NS)].sum(), ':' ,color='C0')
    plt.plot(np.ones(len(sol[i]))*A_lim[NS[0]:NS[1]+NS[0],i+sum(NS)].sum(), ':' ,color='C1')
    plt.plot(np.ones(len(sol[i]))*A_lim[NS[1]+NS[0]:sum(NS),i+sum(NS)].sum(), ':' ,color='C2')
    if i==4-1:
        plt.plot([sol[i][x][0] for x in range(len(sol[0]))], color = 'C0', label='$s=1$')
        plt.plot([sol[i][x][1] for x in range(len(sol[0]))], color = 'C1', label='$s=2$')
        plt.plot([sol[i][x][2] for x in range(len(sol[0]))], color = 'C2', label='$s=3$')
    else:
        plt.plot([sol[i][x][0] for x in range(len(sol[0]))], color = 'C0')
        plt.plot([sol[i][x][1] for x in range(len(sol[0]))], color = 'C1')
        plt.plot([sol[i][x][2] for x in range(len(sol[0]))], color = 'C2')
    plt.ylim(0,1)
    plt.xlim(0, 2000)
    plt.title('Agent {}'.format(i+1+sum(NS)), fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('$i$', fontsize=16)
    plt.ylabel(r'$\widehat x_{{s{i}}}$'.format(i=i+sum(NS)+1), fontsize=16)
    if i==0:
        plt.annotate('%0.2f' % A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum()), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % (A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum()), xy=(1.01,A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum()-0.04), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum()), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
    elif i==1:
        plt.annotate('%0.2f' % (A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum()+.02), xy=(1.01,A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum()+.1), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % (A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum()-.01), xy=(1.01,A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum()-.04), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum()-0.05), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
    elif i==2:
        plt.annotate('%0.2f' % (A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum()), xy=(1.01,A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum()), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % (A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum()-.01), xy=(1.01,A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum()-.04), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum()-0.05), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
    elif i==3:
        plt.annotate('%0.2f' % A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[0*NS[0]:0*NS[0]+NS[0],i+sum(NS)].sum()), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[1*NS[0]:1*NS[0]+NS[0],i+sum(NS)].sum()), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
        plt.annotate('%0.2f' % A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[2*NS[0]:2*NS[0]+NS[0],i+sum(NS)].sum()-.12), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
    else:
        for iv, s in enumerate(sol[i].T):
            plt.annotate('%0.2f' % A_lim[iv*NS[0]:iv*NS[0]+NS[0],i+sum(NS)].sum(), xy=(1.01,A_lim[iv*NS[0]:iv*NS[0]+NS[0],i+sum(NS)].sum()), xycoords=('axes fraction', 'data'), color = 'r', fontsize=13)
plt.figlegend(ncol=3, bbox_to_anchor=(0.45, -.38, 0.5, 0.5), fontsize=16, handlelength=1)
plt.tight_layout()
plt.subplots_adjust(bottom=.2,wspace=.7, right=.98)
plt.savefig(FIG_PATH+'fig7_panel3.pdf', bbox_inches='tight')
#%%
plt.figure(figsize=(5,4.5))
for i in range(4):
    plt.subplot(2,2,i+1)
    h=plt.plot([MU[k][i+sum(NS),:] for k in range(N_PLOT)] )
    plt.title('Agent {}'.format(i+1+sum(NS)), fontsize=16)
    plt.xlabel('$i$', fontsize=16)
    plt.ylabel(r'$\mu_{{{k},i}}(\theta)$'.format(k=i+1+sum(NS)), fontsize=16)
    plt.xticks([0, 100, 200, 300],fontsize=13)
    plt.yticks(fontsize=13)
plt.figlegend(h,[r'$\theta=1$',r'$\theta=2$',r'$\theta=3$'],ncol=4, bbox_to_anchor=(0.45, -.38, 0.5, 0.5), fontsize=16, handlelength=1)
plt.tight_layout()
plt.subplots_adjust(bottom=.2,wspace=.45, right=.98)
plt.savefig(FIG_PATH+'fig7_panel2.pdf', bbox_inches='tight')
