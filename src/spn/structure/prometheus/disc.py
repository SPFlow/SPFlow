import numpy as np
import spn.structure.prometheus.nodes as nd
import networkx as nx
import sys
from spn.structure.prometheus.nodes import *
from spn.structure.prometheus.data import *
from sklearn.metrics import adjusted_mutual_info_score as ami
from time import time
from scipy.stats import multivariate_normal as mn
from sklearn import metrics
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

# Make an affinity matrix. For the datasets of interest, vectorization usually offers speedups that are ~ .1s, so while subpar, this is still fine.

def infmat(mat, nvar):
    retmat = np.zeros((nvar, nvar))
    for i in range(0, nvar):
        for j in range(0, nvar):
            if (i > j):
                retmat[i][j] = retmat[j][i]
            else:
                temp = np.corrcoef(retmat[:, i], retmat[:, j])
                retmat[i][j] = abs(temp[0][1])
    return retmat

# This function makes boolean leaves ( simply, counters ) with a smoothing term. That is, we learn a pdf over all possible binary strings of length nvar


def createpdf(mat, nsam, nvar, smooth=0.05):
    length = int(np.rint(np.power(2, nvar)))
    pdf = np.zeros(length)
    for i in range(0, nsam):
        idx = bintodec(mat[i, :])
        pdf[idx] += float((1.0 - smooth) / nsam)
    for i in range(0, length):
        pdf[i] += float(smooth / float(length))
    return pdf

#decomp takes a graph G. It creates a MST over G, and begins cutting off edges. Now, at every step, there is a check to see if the maximum size CC is below a threshold maxsize, if so, this partition is added to the list of valid partitions. There will be <= maxcount such partitions created as candidate children of sumnodes.


def decomp(G, maxsize, maxcount):
    T = nx.minimum_spanning_tree(G)
    Order = np.asarray(list(T.edges(data='weight')))
    k = len(Order)
    Order = Order[Order[:, 2].argsort()]
    # print(Order)
    Dec = []
    Gc = max(nx.connected_component_subgraphs(T), key=len)
    n = Gc.number_of_nodes()
    if(n <= maxsize):
        Dec.append(list(nx.connected_components(T)))

    count = 0

    for i in range(0, k):
        if(count > maxcount):
            break
        idx, idx2 = int(Order[len(Order) - i - 1, 0]
                        ), int(Order[len(Order) - i - 1, 1])
        T.remove_edge(idx, idx2)
        Gc = max(nx.connected_component_subgraphs(T), key=len)
        n = Gc.number_of_nodes()
        if(n <= maxsize):
            Dec.append(list(nx.connected_components(T)))
            count += 1

    effwts = np.ones(len(Dec)) * (1. / len(Dec))
    s = sumNode()
    s.setwts(effwts)
    return s, Dec

#discmaker is a wrapper that creates a discrete leaf. `Discrete' is a confusing term here, we do not tackle non-binary data.

def discmaker(tempdat, sub):
    l = discNode()
    l.scope = sub
    pdf = createpdf(tempdat[:, sorted(list(sub))], len(tempdat), len(sub))
    l.create(pdf)
    return l


#makenodes is a general wrapper that attempts clustering of the dataset, but the clustering is valid only when the smallest cluster is above some parameter cutoff. We then create recursive calls to either discinduce or continduce, which are as the names suggest called based on the flag passed. The parameters metacutoff determine whether the dataset is big enough to cluster, tempdat is the dataset, scope is the scope, maxsize is the max size of a CC, and if the dataset has scope size falling below indsize ( induce size ) - we pass calls to make leaves.

def makenodes(
        tempdat,
        scope,
        cutoff,
        flag,
        maxsize,
        indsize,
        maxcount,
        metacutoff):
    full = len(tempdat)
    tempdat2 = split(tempdat, 2, scope)
    s = sumNode()
    arr = []
    cnt = 0
    for i in range(0, len(tempdat2)):
        if(len(tempdat2[i]) >= cutoff):
            arr.append(len(tempdat2[i]))
            if(flag == 0):
                s.children.append(
                    discinduce(
                        np.asarray(
                            tempdat2[i]),
                        maxsize,
                        scope,
                        indsize,
                        1,
                        maxcount,
                        cutoff,
                        metacutoff))
            else:
                cutmult, metacutmult = cutoff, metacutoff
                s.children.append(
                    continduce(
                        np.asarray(
                            tempdat2[i]),
                        maxsize,
                        scope,
                        indsize,
                        1,
                        maxcount,
                        cutmult,
                        metacutmult))
            cnt += 1

    for i in range(0, cnt):
        chosen = s.children[i]
        w = 0
        for j in chosen.children:
            s.children.append(j)
            arr.append(chosen.wts[w] * arr[i])
            w += 1
    arr = arr[cnt:]
    s.children = s.children[cnt:]
    s.setwts(arr)
    return s

#This function is the discrete(binary) equivalent to continduce below. Both of them share the function makenodes, which is called when their datasets are large enough to cluster ( i.e. above the parameter metacutoff ). Parameters align with makenodes description above. However, flag is different - flag being 0 signifies attempting a clustering, whereas for makenodes flag determines which one of continduce or discinduce called it.


def discinduce(
        tempdat,
        maxsize,
        scope,
        indsize,
        flag,
        maxcount,
        cutoff=1000,
        metacutoff=3000):
    if (flag == 0):
        if (len(tempdat) >= metacutoff):
            s = makenodes(
                tempdat,
                scope,
                cutoff,
                0,
                maxsize,
                indsize,
                maxcount,
                metacutoff)
            return s

    effdat = eff(tempdat, scope)
    fisher = infmat(effdat, len(scope))
    G = nx.from_numpy_matrix(-abs(fisher))
    G = G.to_undirected()

    s, Dec = decomp(G, maxsize, maxcount)

    for i in range(0, len(Dec)):
        if(len(Dec[i]) > 1):
            p = prodNode()
            s.children.append(p)
            for j in (Dec[i]):
                sub = returnarr(j, scope)
                if (len(j) <= indsize):
                    p.children.append(discmaker(tempdat, sub))
                else:
                    p.children.append(
                        discinduce(
                            tempdat,
                            decrease(maxsize),
                            sub,
                            indsize,
                            0,
                            maxcount,
                            cutoff,
                            metacutoff))

    if(len(scope) <= indsize):
        s.children.append(discmaker(tempdat, sub))

    return s

#Makes a continuous leaf. Counterpart to discmaker.

def contmaker(empmean, effcov, sub, j):
    l = leafNode()
    tempmean = submean(empmean, j)
    tempcov = submat(effcov, j)
    l.scope = sub
    l.create(tempmean, tempcov)
    return l


#Wrapper to induce if the data is continuous. This interacts with makenodes if the size of the dataset exceeds the parameter metacutoff, which basically means the dataset is large enough to warrant clustering. Makenodes is shared with discinduce above, and parameters here are as above except flag which when 0 means a calls to makenodes ( which has a different usage for its own flag )

def continduce(
        tempdat,
        maxsize,
        scope,
        indsize,
        flag,
        maxcount,
        cutmult=3,
        metacutmult=20):
    cutoff = cutmult * (int(np.sqrt(len(scope))))
    metacutoff = metacutmult * (int(np.sqrt(len(scope))))
    if (flag == 0):
        if (len(tempdat) >= metacutoff):
            s = makenodes(
                tempdat,
                scope,
                cutoff,
                1,
                maxsize,
                indsize,
                maxcount,
                metacutoff)
            return s

    effdat = eff(tempdat, scope)
    effcorr = np.corrcoef(np.transpose(effdat))
    effcov = np.cov(np.transpose(effdat))
    empmean = np.mean(effdat, axis=0)
    G = nx.from_numpy_matrix(-abs(effcorr))
    G = G.to_undirected()

    s, Dec = decomp(G, maxsize, maxcount)

    for i in range(0, len(Dec)):
        p = prodNode()
        s.children.append(p)
        for j in (Dec[i]):
            sub = returnarr(j, scope)
            if (len(j) <= indsize):
                p.children.append(contmaker(empmean, effcov, sub, j))
            else:
                p.children.append(
                    continduce(
                        tempdat,
                        decrease(maxsize),
                        sub,
                        indsize,
                        0,
                        maxcount,
                        cutmult,
                        metacutmult))

    return s

# Call to decrease. This is invoked on the maxsize parameter that governs maximum size of a valid CC after a step.
# Note : It is possible to set this in a way that causes the program to bug out. For instance, suppose your maxsize is 9 and you decrement by 4 with a leafsize of 1. This is fine, and will never bug out, since it'll always hit maxsize = 1 after two decrements. But, suppose this was done with decrements of six - it is possible to hit 3 and then -3 which will cause bugouts. For this reason, set the decrement operator to something like a halving ( for univariate leaves ) that always passes through leafsize.
# If the above condition is not followed strictly, it is possible to get
# segfaults running CCCP. Heavily recommended to stick to this


def decrease(value):
    if(value > 7):
        return(max(7, value // 2))
    else:
        return(value - 3)
    # return(value//2)
    # return(value-4) # this is used for CA and SD, our two large continuous
    # datasets. For smaller ones, value-2 runs fast enough


#Code to convert the string we get from Prometheus to a format readable by SPFLOW's txt-to-spn features.

def dumpspnflow(node, flag):
    if(node.kind == 2):
        if(flag == 0):
            return "Categorical(V" + str(list(node.scope)[0]) + "|p=" + (
                np.array2string(node.arr, separator=', ')) + ")"
        else:
            if(len(node.scope) == 1):
                return "Gaussian(V" + str(list(node.scope)[0]) + "|mean=" + str(
                    node.mean[0]) + ";stdev=" + str(np.sqrt(node.cov[0][0])) + ")"
            else:
                fname = "V" + str(list(node.scope)[0])
                for i in range(1, len(node.scope)):
                    fname += "V" + str(list(node.scope)[i])
                cov = node.cov.flatten()
                params = np.hstack((node.mean, cov))
                params = np.array2string(
                    params,
                    separator=',',
                    precision=5).replace(
                    '\n',
                    '')
                return ("MultivariateGaussian(%s|prmset=%s)" % (fname, params))
    elif(node.kind == 1):
        def sumfmt(w, c): return str(w) + "*(" + dumpspnflow(c, flag) + ")"
        children_strs = map(lambda i: sumfmt(
            node.wts[i], node.children[i]), range(len(node.children)))
        return "(" + " + ".join(children_strs) + ")"
    else:
        children_strs = map(
            lambda child: dumpspnflow(
                child, flag), node.children)
        return "(" + " * ".join(children_strs) + ")"


#The main function. We have dset, the dataset, cflag which is 1 for continuous 0 for binary(discrete). Leafsize is the maximum #vars in a leaf. Maxsize indicates the initial upper bound on size of CCs. Maxprods shows how many products at most are created as children of a node. A basic EM step using the scipy objects is run for itermult*#instances over the dataset to return parameters that are not totally meaningless at initialization.

def prometheus(dset, cflag, leafsize=1, maxsize=16, maxprods=8, itermult=1):
    if(cflag == 0):
        msize = maxsize
        width = maxprods
        nlt = dset
        pholder, d = np.shape(dset)
        s = set(range(d))
        #cutoff,metacutoff = _,_
        Tst = discinduce(nlt, msize, s, 1, 0, width)
        Tst.normalize()
        Tst.truncate()
        Strtodump = dumpspnflow(Tst, 0)
        return Strtodump
    else:
        pholder, d = np.shape(dset)
        s = set(range(d))
        ab = dset
        #ab = whiten(np.asarray(ab))
        Tst = continduce(ab, maxsize, s, leafsize, 0, maxprods)
        Tst.normalize()
        Tst.truncate()
        length = len(ab)
        for i in range(0, itermult * length):
            t = time()
            idx = np.random.randint(0, length)
            nd.globalarr = ab[idx]
            Tst.passon()
            placeholder = Tst.retval()
            Tst.update()
            print("Iteration timestep", time() - t)
        Tst.normalize()
        Strtodump = dumpspnflow(Tst, 1)
        return Strtodump
