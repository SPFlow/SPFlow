#Some helper functions to make the entire thing run
import nodes
import math
import numpy as np
import scipy
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.vq import vq, kmeans, whiten

#Converts array indices from function-level indices to global-level objective indices. For example, consider the set [0,1,2,3] and suppose you have passed the [1,3]'s into a call. They will be treated as [0,1] and you will convert them back.
#The usage of the set functionality slows up the implementation a bit and is not strictly necessary. However, it's a good idea to keep them this way since this simplifies the code. 

def eff(tempdat,scope):
	effdat = np.zeros((len(tempdat),len(scope)))
	for i in range(0,len(tempdat)):
		temp = submean(tempdat[i],scope)
		for j in range(0,len(scope)):
			effdat[i][j] = temp[j]
	return effdat

def returnarr(arr,scope):
	q = []
	te = list(scope)
	te = sorted(te)
	for i in arr:
		q.append(te[i])
	return set(q)

def split(arr,k,scope):
	pholder,clusters = scipy.cluster.vq.kmeans2(arr[:,sorted(list(scope))],k,minit='points')
	print ("clusters",clusters)
	big = []
	for i in range(0,len(set(clusters))):
		small = []
		for j in range(0,len(arr)):
			if (clusters[j]==i):
				small.append(arr[j,:])
		big.append(small)
		#print(big)
	return big

def submat(mat,subset):
	q = len(subset)
	print(q)
	ret = np.zeros((q,q))
	w = 0
	for i in subset:
		z = 0
		for j in subset:
			ret[w,z] = mat[i,j] 
			z+=1
		w+=1
	return ret

def submean(mean,subset):
	q = len(subset)
	m = np.zeros(q)
	w = 0
	for i in subset:
		m[w] = mean[i]
		w+=1
	return m

