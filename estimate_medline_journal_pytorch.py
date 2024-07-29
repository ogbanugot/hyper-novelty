import time
import pandas as pd
import scipy.sparse as ssp
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.stats import dirichlet
from collections import Counter
from datetime import datetime
import random
from collections import defaultdict
from scipy.optimize import minimize
from itertools import zip_longest
from numba import guvectorize,vectorize
from numba import int64,float64,int32
from numba import cuda
import math
import pickle as pickle
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve
from functools import partial
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import pylab as plt
import seaborn as sns
import torch
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

class Stopwatch:
    start_time=None
    def go(self,msg=''):
        if msg:
            print(msg, flush=True)
        self.start_time=time.time()
    def stop(self,msg=''):
        if msg:
            print("{}: {} seconds".format(msg,time.time()-self.start_time), flush=True)
        else:
            print("Elapsed time: {} seconds".format(time.time()-self.start_time), flush=True)
    def check(self):
        return time.time()-self.start_time

tic=Stopwatch()

def load_date(filename):
    tic=Stopwatch()
    print("Loading paper dates %s from disk..." % filename),
    tic.go()
    pkl_file = open(filename, 'rb')
    A=pickle.load(pkl_file,encoding='latin1')
    pkl_file.close()
    tic.stop()
    return A

def load_hypergraph(filename):
    tic=Stopwatch()
    print("Loading file %s from disk..." % filename),
    tic.go()
    pkl_file = open(filename, 'rb')
    (row,col) = pickle.load(pkl_file,encoding='latin1')
    pkl_file.close()
    A=ssp.coo_matrix((np.ones(len(row),dtype=np.int8),(row,col)),shape=(19916562, max(col)+1),dtype=np.int8)
    tic.stop()
    return A

tic=Stopwatch()
G=load_hypergraph('../data/medline/journals.pkl').tocsr()
paper_dates=load_date('../data/medline/paper_dates.pkl')

G=G[paper_dates>0,:]
paper_dates=paper_dates[paper_dates>0]
G=G[paper_dates<2010,:]
paper_dates=paper_dates[paper_dates<2010]
paper_dates[paper_dates<1950]=1950

def get_nodes(e, G0):
    # Convert sparse row to dense array and get non-zero indices
    dense_row = G0[e, :].toarray().flatten()
    return tuple(sorted(np.nonzero(dense_row)[0]))

def worker_function(e, G0):
    # This function is now at the module level and can be pickled
    return get_nodes(e, G0)

def get_hyperedges(G0):
    # Convert sparse matrix to dense array for comparison
    dense_G0 = G0.toarray()
    x = (dense_G0.sum(axis=1) > 1).flatten()
    
    if len(x) == 0:
        return Counter()
    
    indices = np.nonzero(x)[0]

    # Create a partial function to pass additional arguments
    partial_worker = partial(worker_function, G0=G0)

    with Pool(cpu_count() - 1) as pool:
        E = Counter(pool.map(partial_worker, indices))
    
    return E

def batch_generator(iterable, batchsize, shuffle=False):
    if shuffle:
        iterable = list(iterable)
        random.shuffle(iterable)
    sentinel = object()
    return ([entry for entry in i if entry is not sentinel]
            for i in zip_longest(*([iter(iterable)] * batchsize), fillvalue=sentinel))


def edges2CSR(H):
    nodes = []
    nodes_in = [0]
    for h in H:
        nodes += list(h)
        nodes_in.append(len(nodes))
    return nodes, nodes_in

def CSR2CSC(nodes, nodes_in, N):
    M = csr_matrix((np.ones_like(nodes, dtype=np.int8), nodes, nodes_in), shape=(len(nodes_in)-1, N))
    M = M.tocsc()
    return M.indices, M.indptr

def dtheta(x, theta, active, edges, edges_in, nodes, nodes_in, weights, rand_edges, rand_edges_in, rand_nodes, rand_nodes_in):
    K = theta.shape[1]
    i = x // K
    z = x % K
    res = torch.zeros(1, device=theta.device)
    for j in edges[edges_in[i]:edges_in[i+1]]:
        dlam = 1.0
        for u in nodes[nodes_in[j]:nodes_in[j+1]]:
            if u != i:
                dlam *= theta[active[u], z]
        p = 0.0
        for k in range(K):
            prod = 1.0
            for u in nodes[nodes_in[j]:nodes_in[j+1]]:
                prod *= theta[active[u], k]
            p += prod
        p = max(p, 1e-8)
        res += weights[j] * dlam / p - dlam

    for j in rand_edges[rand_edges_in[i]:rand_edges_in[i+1]]:
        dlam = 1.0
        for u in rand_nodes[rand_nodes_in[j]:rand_nodes_in[j+1]]:
            if u != i:
                dlam *= theta[active[u], z]
        res -= dlam

    return res

def preupdate(theta, theta0, r, s):
    with torch.no_grad():
        theta -= 1.0 / (100 + r) * (theta - theta0) / s

def update(theta, active, D, r):
    with torch.no_grad():
        theta[active] += 1.0 / (100 + r) * D
        torch.clamp(theta, min=0.001, out=theta)

def p_pos(i, theta, active, nodes, nodes_in, weights):
    p = 0.0
    K = theta.shape[1]
    for k in range(K):
        prod = 1.0
        for u in nodes[nodes_in[i]:nodes_in[i+1]]:
            prod *= theta[active[u], k]
        p += prod
    return weights[i] * torch.log(torch.tensor(p, device=theta.device)) - p if p > 1e-15 else torch.tensor(0.0, device=theta.device)

def p_neg(i, theta, active, rand_nodes, rand_nodes_in):
    p = 0.0
    K = theta.shape[1]
    for k in range(K):
        prod = 1.0
        for u in rand_nodes[rand_nodes_in[i]:rand_nodes_in[i+1]]:
            prod *= theta[active[u], k]
        p += prod
    return -p

def logPG(theta, active, edges, edges_in, nodes, nodes_in, weights, rand_edges, rand_edges_in, rand_nodes, rand_nodes_in):
    res1 = torch.stack([p_pos(i, theta, active, nodes, nodes_in, weights) for i in range(weights.shape[0])])
    sum1 = res1.sum().item()
    res2 = torch.stack([p_neg(i, theta, active, rand_nodes, rand_nodes_in) for i in range(rand_nodes_in.shape[0] - 1)])
    sum2 = res2.sum().item()
    return (sum1 + sum2)

def estimate(G, times, K=20, thetas=None, nepochs=5, subepochs=10, batchsize=1000, discontinue=1, outfile=None):
    tic.go('Estimating...')
    candidate_times = np.unique(times)
    if thetas is None:
        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(K) * 0.5)
        theta = dirichlet_dist.sample((G.shape[1],)).cuda()
        thetas = [theta]
    else:
        if K != thetas[0].shape[1]:
            raise ValueError("K and the dimension of initial condition don't match!")
        thetas = [thetas[0]] + thetas
        thetas = [t.cuda() for t in thetas]

    threadsperblock = (32, min(32, K))
    blockspergrid_x = math.ceil(thetas[0].shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(K / threadsperblock[1])
    blockspergrid0 = (blockspergrid_x, blockspergrid_y)

    likelihoods = []
    for epoch in range(nepochs):
        print(f"epoch {epoch} of {nepochs}")
        time_index = 1
        # candidate_times = [candidate_times[0]]
        # print(candidate_times)
        for t in candidate_times:
            if len(thetas) == time_index:
                thetas.append(thetas[-1].clone())
            G0 = G[(times == t).nonzero()[0], :]
            active_nodes=G0.sum(axis=0).A.ravel().nonzero()[0]
            G0 = G0[:, active_nodes]
            G0.data=np.ones_like(G0.data)
            E0 = get_hyperedges(G0)
            N = len(active_nodes)
            V = range(N)
            active_nodes_tensor = torch.tensor(active_nodes)
            active_nodes_tensor = active_nodes_tensor.cuda()

            blockspergrid_x = math.ceil(N / threadsperblock[0])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            if outfile:
                outfile.write(f"Epoch {epoch} | Time: {t} | Nodes: {N} | Edges: {len(E0)} | ")
                outfile.flush()

            for subepoch in range(subepochs):
                print(f"subepoch {subepoch} of {subepochs}")
                batch_indx = 0
                # for batch in batch_generator(E0.items(), batchsize):
                for batch_indx, batch in enumerate(batch_generator(E0.items(), batchsize)):
                    # print(f"batch {batch_indx} of {batchsize}")
                    print(f"batch {batch_indx} of {len(E0.items()) // batchsize + 1}")
                    samples, weights = zip(*batch)
                    nodes, nodes_in = edges2CSR(samples)
                    edges, edges_in = CSR2CSC(nodes, nodes_in, N)
                    nodes = torch.tensor(nodes, dtype=torch.int32).cuda()
                    nodes_in = torch.tensor(nodes_in, dtype=torch.int32).cuda()
                    weights = torch.tensor(weights, dtype=torch.float32).cuda()
                    edges = torch.tensor(edges, dtype=torch.int32).cuda()
                    edges_in = torch.tensor(edges_in, dtype=torch.int32).cuda()

                    E_neg = []
                    h_indx = 0
                    for h in samples:
                        while True:
                            e = tuple(sorted(np.random.choice(V, len(h), replace=False)))
                            if e not in E0:
                                E_neg.append(e)
                                break
                        h_indx += 1
                    rand_nodes, rand_nodes_in = edges2CSR(E_neg)
                    rand_edges, rand_edges_in = CSR2CSC(rand_nodes, rand_nodes_in, N)
                    rand_nodes = torch.tensor(rand_nodes, dtype=torch.int32).cuda()
                    rand_nodes_in = torch.tensor(rand_nodes_in, dtype=torch.int32).cuda()
                    rand_edges = torch.tensor(rand_edges, dtype=torch.int32).cuda()
                    rand_edges_in = torch.tensor(rand_edges_in, dtype=torch.int32).cuda()

                    preupdate(thetas[time_index], thetas[time_index-1], (epoch + 1) * (subepoch + 1), discontinue)
                    D = torch.stack([dtheta(x, thetas[time_index], active_nodes, edges, edges_in, nodes, nodes_in, weights, rand_edges, rand_edges_in, rand_nodes, rand_nodes_in) for x in range(N * K)])
                    D = D.view(N, K)
                    update(thetas[time_index], active_nodes, D, (epoch + 1) * (subepoch + 1))
                    l = logPG(thetas[time_index], active_nodes, edges, edges_in, nodes, nodes_in, weights, rand_edges, rand_edges_in, rand_nodes, rand_nodes_in)
                    likelihoods.append(l / len(samples))
                    batch_indx += 1

            if outfile:
                outfile.write('log-lik: {:.2f} | Elapsed {:.2f} s\n'.format(likelihoods[-1], tic.check()))
                outfile.flush()
            time_index+=1
        thetas[0]=thetas[1]
    tic.stop()
    return (thetas[1:], likelihoods)

print("{}, # nodes: {} | # edges: {} | average degree: {} | average edge size: {}".format(datetime.now(),G.shape[1],G.shape[0],G.sum(axis=0).mean(),G.sum(axis=1).mean()))

K = 20 # dimension of hidden space

with open("output_medline_journal.txt", 'w') as logfile:
    thetas,likelihoods=estimate(G,paper_dates,K,discontinue=1.0,nepochs=5,batchsize=2000,outfile=logfile)

thetas=[theta.copy_to_host() for theta in thetas]

with open('../Fitted_Model/block_model_medline_journal.pkl','wb') as outfile2:
    pickle.dump([thetas,likelihoods,np.unique(paper_dates)],outfile2)