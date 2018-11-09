import numpy as np
from data import *
from scipy.stats import multivariate_normal as mn

globalarr = []

def convert(tag):
	holder = ['PRD','SUM','BINNODE']
	return holder[tag]

def returnlg(wts,vals):
	base = np.amax(vals)
	wtsum = 0
	for i in range(0,len(vals)):
		vals[i] = vals[i] - base
		wtsum += wts[i]
	sum = 0
	for i in range(0,len(vals)):
		sum += wts[i]*(np.exp(vals[i]))
	return(np.log(sum) + base - np.log(wtsum))

def bintodec(arr):
	wt = np.rint(np.power(2,len(arr)-1))
	cnt = 0
	for i in range(0,len(arr)):
		cnt += wt*arr[i]
		wt = wt/2
	return int(np.rint(cnt))


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
		for i in range(0,childcnt):
			chosen = self.children[i]
			if(chosen.kind==2):
				continue
			else:
				grandkids = len(chosen.children)
				if (grandkids==1):
					dele.append(i)
					grandkid = chosen.children[0]		
					for j in grandkid.children:
						self.children.append(j)
		for idx in sorted(dele,reverse=True):
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

	
	def setwts(self,arr):
		for i in arr:
			self.wts.append(i)

	def truncate(self):
		for i in self.children:
			i.truncate()			

	def normalize(self):
		sum = 1e-11
		for i in range(0,len(self.wts)):
			sum += self.wts[i]
		for i in range(0,len(self.wts)):
			self.wts[i] = self.wts[i]/sum
		for j in self.children:
			j.normalize()

	def retval(self):
		arr = []
		for i in self.children:
			arr.append(i.retval())
		self.value = returnlg(self.wts,arr)
		#print(self.value)
		return (self.value)

	def update(self):
		inf = -1e19
		j = 0
		#winnode = self.children[j]
		winidx = 0
		for i in self.children:
			if((i.value)>inf):
				inf = (i.value)
				winnode = i
				winidx = j
			j = j+1
		self.wts[winidx] = self.wts[winidx]+1
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
		
	def create(self,mean,cov):
		try:
			self.pdf = mn(mean=mean,cov=cov)
			self.mean = mean
			self.cov = cov
		except:
			cov[np.diag_indices_from(cov)] += 1e-4
			self.pdf = mn(mean=mean,cov=cov)
			self.mean = mean
			self.cov = cov

	def passon(self):
		self.rec = submean(globalarr,self.scope)
		self.value = self.pdf.logpdf(self.rec)

	def retval(self):
		return(self.value)
	
	def update(self):
		
		tempmean = np.zeros(len(self.mean))
		for i in range(0,len(self.mean)):
			tempmean[i] = self.mean[i] + float((self.rec[i] - self.mean[i])/(float(self.counter)))
		for i in range(0,len(self.mean)):
			for j in range(0,len(self.mean)):
				self.cov[i][j] = float((self.cov[i][j]*(self.counter-1) + (self.rec[i]-tempmean[i])*(self.rec[j] - self.mean[j]))/(self.counter))
		self.mean = tempmean
		try:
			self.pdf = mn(mean=self.mean,cov=self.cov)
		except:
			self.cov[np.diag_indices_from(self.cov)] += 1e-4
			self.pdf = mn(mean=self.mean,cov=self.cov)
		self.counter = self.counter+1
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

	def create(self,pdfarr):
		self.arr = pdfarr
		self.size = len(pdfarr)

	def passon(self):
		self.rec = submean(globalarr,self.scope)
		self.value = np.log(self.arr[bintodec(self.rec)])

	def retval(self):
		return (self.value)

	def update(self):
		idx = bintodec(self.rec)
		self.arr[idx] = float(self.arr[idx]) + float((1.0)/float(self.counter))
		for i in range(0,self.size):
			self.arr[i] = float(float(self.arr[i])/(1.0+float((1.0)/float(self.counter))))
		self.counter = self.counter+1
		


