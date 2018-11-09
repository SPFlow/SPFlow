import numpy as np
import nodes as nd
import networkx as nx
import sys
from nodes import *
from data import *
from sklearn.metrics import adjusted_mutual_info_score as ami
from time import time
from scipy.stats import multivariate_normal as mn
from sklearn import metrics
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

#Make the affinity matrix

def infmat(mat,nvar):
	retmat = np.zeros(nvar*nvar)
	retmat = np.reshape(retmat,(nvar,nvar))
	for i in range(0,nvar):
		for j in range(0,nvar):
			if (i>j):
				retmat[i][j] = retmat[j][i]
			else:
				retmat[i][j] = (float(np.dot(mat[:,i],mat[:,j])*np.dot(mat[:,i],mat[:,j]) + 1e-4))/(float(np.dot(mat[:,i],mat[:,i])*np.dot(mat[:,j],mat[:,j]) + 1e-2))
				#temp = np.corrcoef(retmat[:,i],retmat[:,j])
				#retmat[i][j] = abs(temp[0][1])
	return retmat

#Get boolean leaves

def createpdf(mat,nsam,nvar):
	length = int(np.rint(np.power(2,nvar)))
	pdf = np.zeros(length)
	for i in range(0,nsam):
		#print(mat[i,:])
		idx = bintodec(mat[i,:])
		#print(idx)
		pdf[idx] += float((0.95)/nsam)
	for i in range(0,length):
		pdf[i] += float(0.05/float(length))
	return pdf

def decomp(G,maxsize,maxcount):
	T=nx.minimum_spanning_tree(G)
	Order = np.asarray(T.edges(data='weight'))
	k = len(Order)
	Order = Order[Order[:,2].argsort()]
	Dec = []
	Gc = max(nx.connected_component_subgraphs(T), key=len)
	n = Gc.number_of_nodes()
	if(n<=maxsize):
		Dec.append(list(nx.connected_components(T)))

	count = 0
	
	for i in range(0,k):
		if(count>maxcount):
			break
		idx = int(Order[len(Order)-i-1,0])
		idx2 = int(Order[len(Order)-i-1,1])
		T.remove_edge(idx,idx2)
		Gc = max(nx.connected_component_subgraphs(T), key=len)
		n = Gc.number_of_nodes()
		if(n<=maxsize):
			Dec.append(list(nx.connected_components(T)))
			count += 1

	effwts = np.zeros(len(Dec))
	for i in range(0,len(Dec)):
		effwts[i] = 1./len(Dec)

	s = sumNode()
	s.setwts(effwts)
	return s,Dec

def discmaker(tempdat,sub):
	l = discNode()
	l.scope = sub
	pdf = createpdf(tempdat[:,sorted(list(sub))],len(tempdat),len(sub))
	l.create(pdf)
	return l

def makenodes(tempdat,scope,cutoff,flag,maxsize,indsize,maxcount,metacutoff):
		full = len(tempdat)
		tempdat2 = split(tempdat,2,scope)
		s = sumNode()
		arr = []
		cnt = 0
		for i in range(0,len(tempdat2)):
			if(len(tempdat2[i])>=cutoff):
				arr.append(len(tempdat2[i]))
				if(flag==0):
					s.children.append(discinduce(np.asarray(tempdat2[i]),maxsize,scope,indsize,1,maxcount,cutoff,metacutoff))
				else:
					cutmult,metacutmult = cutoff/len(scope),metacutoff/len(scope)
					s.children.append(continduce(np.asarray(tempdat2[i]),maxsize,scope,indsize,1,maxcount,cutmult,metacutmult))
				cnt += 1
		
		for i in range(0,cnt):
			chosen = s.children[i]
			w = 0
			for j in chosen.children:
				s.children.append(j)
				arr.append(chosen.wts[w]*arr[i])
				w += 1
		arr = arr[cnt:]
		s.children = s.children[cnt:]			
		s.setwts(arr)
		return s


def discinduce(tempdat,maxsize,scope,indsize,flag,maxcount,cutoff=1000,metacutoff=3000):
	if (flag==0):
		if (len(tempdat)>=metacutoff):
			s = makenodes(tempdat,scope,cutoff,0,maxsize,indsize,maxcount,metacutoff)
			return s	

	effdat = eff(tempdat,scope)
	fisher = infmat(effdat,len(scope))
	G = nx.from_numpy_matrix(-abs(fisher))
	G = G.to_undirected()

	s,Dec = decomp(G,maxsize,maxcount)

	for i in range(0,len(Dec)):
		if(len(Dec[i])>1):	
			p = prodNode()
			s.children.append(p)
			for j in (Dec[i]):
				sub = returnarr(j,scope)
				if (len(j)<=indsize):
					p.children.append(discmaker(tempdat,sub))
				else:
					p.children.append(discinduce(tempdat,decrease(maxsize),sub,indsize,0,maxcount,cutoff,metacutoff))
	
	if(len(scope)<=indsize):
		s.children.append(discmaker(tempdat,sub))

	return s

def contmaker(empmean,effcov,sub,j):
	l = leafNode()
	tempmean = submean(empmean,j)
	tempcov = submat(effcov,j)
	l.scope = sub
	l.create(tempmean,tempcov)
	return l
	

def continduce(tempdat,maxsize,scope,indsize,flag,maxcount,cutmult=1,metacutmult=10):
	cutoff = cutmult*len(scope)
	metacutoff = metacutmult*len(scope)
	if (flag==0):
		if (len(tempdat)>=metacutoff):
			s = makenodes(tempdat,scope,cutoff,1,maxsize,indsize,maxcount,metacutoff)
			return s
			
	effdat = eff(tempdat,scope)
	effcorr = np.corrcoef(np.transpose(effdat))
	effcov = np.cov(np.transpose(effdat))
	empmean = np.mean(effdat,axis=0)
	G = nx.from_numpy_matrix(-abs(effcorr))
	G = G.to_undirected()

	s,Dec = decomp(G,maxsize,maxcount)

	for i in range(0,len(Dec)):
		p = prodNode()
		s.children.append(p)
		for j in (Dec[i]):
			sub = returnarr(j,scope)
			if (len(j)<=indsize):
				p.children.append(contmaker(empmean,effcov,sub,j))
			else:
				p.children.append(continduce(tempdat,decrease(maxsize),sub,indsize,0,maxcount,cutmult,metacutmult))
		

	return s

#Call to decrease.
#Note : It is possible to set this in a way that causess the program to bug out. For instance, suppose your maxsize is 9 and you decrement by 4 with a leafsize of 1. This is fine, and will never bug out, since it'll always hit maxsize = 1 after two decrements. But, suppose this was done with decrements of six - it is possible to hit 3 and then -3 which will cause bugouts. For this reason, set the decrement operator to something like a halving ( for univariate leaves ) that always passes through leafsize.
#If the above condition is not followed strictly, it is possible to get segfaults running CCCP. Heavily recommended to stick to this

def decrease(value):
	if(value>7):
		return(max(7,value/4))
	else:
		return(value-2)
	#return(value/2)
	#return(value-4) # this is used for CA and SD, our two large continuous datasets. For smaller ones, value-2 runs fast enough
	
f = sys.argv[1]
d = int(sys.argv[2])
cflag = int(sys.argv[3])

def dumpspnflow(node,flag):
	if(node.kind==2):
		if(flag==0):
			return "Categorical(V"+str(list(node.scope)[0])+"|p="+(np.array2string(node.arr, separator=', '))+")"
		else:
			return "Gaussian(V"+str(list(node.scope)[0])+"|mean="+str(node.mean[0])+";stdev="+str(np.sqrt(node.cov[0][0]))+")"
	elif(node.kind==1):
		def sumfmt(w, c): return str(w) + "*(" + dumpspnflow(c, flag) + ")"
		children_strs = map(lambda i: sumfmt(node.wts[i], node.children[i]), range(len(node.children)))
		return "(" + " + ".join(children_strs) + ")"
	else:
		children_strs = map(lambda child: dumpspnflow(child, flag), node.children)
        return "(" + " * ".join(children_strs) + ")"				

if(cflag==0):
	msize = 16
	width = 8
	nlt = np.genfromtxt(f,delimiter=",")
	
	s = set(xrange(d))
	#cutoff,metacutoff = _,_
	Tst = discinduce(nlt,msize,s,1,0,width)
	Tst.normalize()
	Tst.truncate()
	Strtodump = dumpspnflow(Tst,0)
	file = open('./treestruct.txt', 'w')
	file.write(Strtodump)
	file.close()
else:
	values = []
	s = set(xrange(d))
	ab = np.genfromtxt(f,delimiter=",")
	ab = whiten(np.asarray(ab))
	Tst = continduce(ab,16,s,1,0,4)
	Tst.normalize()
	Tst.truncate()
	Strtodump = dumpspnflow(Tst,1)
	file = open('./treestruct.txt', 'w')
	file.write(Strtodump)
	file.close()



