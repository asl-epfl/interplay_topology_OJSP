import numpy as np
from scipy import stats
from decimal import *

def build_adjacency_metropolis(N, G):
    '''
    Builds a combination matrix using a Metropolis rule.
    N: number of nodes.
    G: Adjacency matrix.
    '''
    A = np.zeros((N, N))
    nk = G.sum(axis=1)
    for k in range(N):
        for l in range(N):
            if G[k,l]==1 and k!=l:
                A[k,l] = 1/np.max([nk[k], nk[l]])
        A[k,k] = 1- A[k].sum()
    return A.T

def build_adjacency_averaging(N, G):
    '''
    Builds a combination matrix using an averaging rule.
    N: number of nodes.
    G: Adjacency matrix.
    '''
    A = np.zeros((N, N))
    nk = G.sum(axis=1)
    for l in range(N):
        for k in range(N):
            if G[l,k]==1:
                A[l,k] = 1/nk[k]
    return A

def gaussian(x, m, var):
    '''
    Computes the Gaussian pdf value at x.
    x: value at which the pdf is computed (Decimal type)
    m: mean (Decimal type)
    var: variance
    '''
    p = np.exp(-(x-m)**2/(2*var))/(np.sqrt(2*np.pi*var))
    return p

def bayesian_update(L, mu):
    '''
    Computes the Bayesian update.
    L: likelihoods matrix.
    mu: beliefs matrix.
    '''
    aux = L*mu
    bu = aux/aux.sum(axis = 1)[:, None]
    return bu

def beta_dist(x, a, b):
    '''
    Computes the Beta pdf value at x.
    x: value at which the pdf is computed (Decimal type)
    a, b: shape parameters
    '''
    p = stats.beta.pdf(x,a,b)
    return p

def build_y(PSI_DEC, M, NS, N_ITER, N_T, NR):
    '''
    Builds a vector y for the linear system Cx=y.
    PSI_DEC: log-beliefs measurements over time
    M: number of hypotheses
    NS: number of agents for each sending network
    N_ITER: number of iterations
    N_T: number of agents
    NR: number of receiving agents
    '''
    psi_dec = [(PSI_DEC[-1][x,[k for k in range(M) if k !=np.argmax(PSI_DEC[-1][sum(NS):],axis=1)[(x-sum(NS))]]]/(N_ITER))[0] for x in range(sum(NS), N_T)]
    psi_dec_const= np.array([[psi_dec[i], Decimal(1) ]for i in range(NR)])
    y = np.array([[float(s) for s in d] for d in psi_dec_const])
    return y

def build_matrix_D(NS, NR, FS, L0, L1, dt):
    '''
    Builds a matrix D of divergences which composes the matrix C in the linear system Cx=y.
    NS: number of agents for each sending component
    NR: number of receiving agents
    FS: true distributions for the sending draw_networkx_edges
    L0, L1: likelihoods for two hypotheses
    dt: step size
    '''
    D_=[]
    D_=np.array([[np.sum(Y*np.log(Y/X)*dt) for X in [L0[iY], L1[iY]]] for iY, Y in enumerate(FS)] ).T
    D = np.array([[float(x) for x in s] for s in D_])
    return D

def build_matrix_C(D_, psivec, NS, NR, FS, L0, L1, M):
    '''
    Builds a matrix C for the linear system Cx=y.
    D_: matrix of divergences
    psivec: vector of log-beliefs
    NS: number of agents for each sending network
    NR: number of receiving agents
    FS: true distributions for the sending draw_networkx_edges
    L0, L1: likelihoods for two hypotheses
    M: number of hypotheses
    '''
    TR_=np.argmax(psivec)
    auxv=np.zeros(M)
    auxv[TR_]=1
    C=((np.ones(2)*auxv[:,None]).T-np.eye(M))@D_
    C=np.delete(C,TR_, axis=0)
    C=np.vstack([C, np.ones(2)])
    return C
