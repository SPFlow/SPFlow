import numpy as np
from spn.structure.prometheus.data import *
from scipy.stats import multivariate_normal as mn

#Ignore this, only useful if you wish to train parameters via scipy

globalarr = []

#Type of node

def convert(tag):
    holder = ['PRD', 'SUM', 'BINNODE']
    return holder[tag]


def returnlg(wts, vals):
    arr1, arr2 = np.asarray(wts), np.asarray(vals)
    base = np.amax(arr2)
    arr2 = arr2 - base
    wtsum = np.sum(arr1[:len(arr2)])
    aux = np.exp(arr2)
    aux = np.sum(np.multiply(aux, arr1))
    return(np.log(aux) + base - np.log(wtsum))


def bintodec(arr):
    wt = np.rint(np.power(2, len(arr) - 1))
    cnt = 0
    for i in range(0, len(arr)):
        cnt += wt * arr[i]
        wt = wt // 2
    return int(np.rint(cnt))


#The entirety of the Node code + the ones that inherit from the class, are basically slightly modified versions of Agastya Kalra's OSLRAU implementation. Since each node is a separate object, beware #of memorysinks ( should not be an issue if running on servers )

class Node:
    def __init__(self):
        self.scope = set()
        self.children = []
        self.value = 0
        self.kind = 0
        self.createid = 0

    def passon(self):
        for i in self.children:
            i.passon()

    def normalize(self):
        for j in self.children:
            j.normalize()


class prodNode(Node):
    def retval(self):
        Logval = 0
        for i in self.children:
            Logval += i.retval()
        self.value = Logval
        return (self.value)

    def update(self):
        for i in self.children:
            i.update()

    def truncate(self):
        childcnt = len(self.children)
        dele = []
        for i in range(0, childcnt):
            chosen = self.children[i]
            if(chosen.kind == 2):
                continue
            else:
                grandkids = len(chosen.children)
                if (grandkids == 1):
                    dele.append(i)
                    grandkid = chosen.children[0]
                    for j in grandkid.children:
                        self.children.append(j)
        for idx in sorted(dele, reverse=True):
            self.children.pop(idx)
        for i in self.children:
            i.truncate()


class sumNode(Node):
    def __init__(self):
        self.scope = []
        self.children = []
        self.wts = []
        self.value = 0
        self.kind = 1

    def setwts(self, arr):
        for i in arr:
            self.wts.append(i)

    def truncate(self):
        for i in self.children:
            i.truncate()

    def normalize(self):
        wtsum = 1e-11 + sum(self.wts)
        self.wts = [x / wtsum for x in self.wts]
        for j in self.children:
            j.normalize()

    def retval(self):
        arr = []
        for i in self.children:
            arr.append(i.retval())
        self.value = returnlg(self.wts, arr)
        # print(self.value)
        return (self.value)

    def update(self):
        inf = -np.inf
        j = 0
        #winnode = self.children[j]
        winidx = 0
        for i in self.children:
            if((i.value) > inf):
                inf = (i.value)
                winnode = i
                winidx = j
            j += 1
        self.wts[winidx] += 1
        winnode.update()


class leafNode(Node):
    def __init__(self):
        self.value = 0
        self.flag = 1
        self.mean = []
        self.cov = []
        self.rec = []
        self.scope = []
        self.counter = 5.0
        self.kind = 2

    def normalize(self):
        return

    def truncate(self):
        return

    def create(self, mean, cov):
        try:
            self.pdf = mn(mean=mean, cov=cov)
            self.mean = mean
            self.cov = cov
        except BaseException:
            cov[np.diag_indices_from(cov)] += 1e-4
            self.pdf = mn(mean=mean, cov=cov)
            self.mean = mean
            self.cov = cov

    def passon(self):
        self.rec = submean(globalarr, self.scope)
        self.value = self.pdf.logpdf(self.rec)

    def retval(self):
        return(self.value)

    def update(self):

        tempmean = np.zeros(len(self.mean))
        tempmean = self.mean + ((self.rec - self.mean) / (float(self.counter)))
        self.cov = ((self.cov * (self.counter - 1.0)) + \
                    (np.outer((self.rec - tempmean), (self.rec - self.mean))) / (self.counter))
        self.mean = tempmean
        try:
            self.pdf = mn(mean=self.mean, cov=self.cov)
        except BaseException:
            self.cov[np.diag_indices_from(self.cov)] += 1e-4
            self.pdf = mn(mean=self.mean, cov=self.cov)
        self.counter += 1.0
        return


class discNode(Node):
    def __init__(self):
        self.value = 0
        self.flag = 1
        self.kind = 2
        self.rec = []
        self.scope = []
        self.arr = []
        self.size = 0
        self.counter = 2.0

    def truncate(self):
        return

    def normalize(self):
        return

    def create(self, pdfarr):
        self.arr = pdfarr
        self.size = len(pdfarr)

    def passon(self):
        self.rec = submean(globalarr, self.scope)
        self.value = np.log(self.arr[bintodec(self.rec)])

    def retval(self):
        return (self.value)

    def update(self):
        idx = bintodec(self.rec)
        self.arr[idx] = float(self.arr[idx]) + \
            float((1.0) / float(self.counter))
        for i in range(0, self.size):
            self.arr[i] = float(float(self.arr[i]) /
                                (1.0 + float((1.0) / float(self.counter))))
        self.counter = self.counter + 1
