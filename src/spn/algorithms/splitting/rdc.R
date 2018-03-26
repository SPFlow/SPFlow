# Title     : TODO
# Objective : TODO
# Created by: molina
# Created on: 3/20/18

library(iterators)
library(parallel)
library(foreach)
library(doMC)
library(dummies)
library(energy)
library(doRNG)
registerDoMC(detectCores()-1)



transformRDC <- function(data, ohe, featureTypes, min_k=10, s=1/6, f=sin) {

  columnnames <- names(data)
  set.seed(123)
  rdctrans <- foreach(i = 1:ncol(data), .combine=cbind) %dorng% {
    x <- data[,i]

    if(ohe && (featureTypes[[i]] == "discrete" || featureTypes[[i]] == "binary")){
      x <- dummy(columnnames[i], data, sep="_")
    }

    x <- cbind(apply(as.matrix(x),2,function(u)ecdf(u)(u)),1)
    k = max(min_k, ncol(x))


    x <- s/ncol(x)*x%*%matrix(rnorm(ncol(x)*k),ncol(x))

    x <- f(x)

    return(x)
  }

  return(rdctrans)
}

rdc <- function(x,y,s=1/6,f=sin, linear=FALSE) {
  if(var(x) == 0|| var(y) == 0) {
    return(0)
  }
  x <- cbind(apply(as.matrix(x),2,function(u)ecdf(u)(u)),1)
  y <- cbind(apply(as.matrix(y),2,function(u)ecdf(u)(u)),1)
  k = max(ncol(x), ncol(y))

  set.seed(42)
  x <- s/ncol(x)*x%*%matrix(rnorm(ncol(x)*k),ncol(x))
  set.seed(43)
  y <- s/ncol(y)*y%*%matrix(rnorm(ncol(y)*k),ncol(y))


  if(linear){
    xy <- cancor(cbind(f(x),1),cbind(f(y),1))$cor[1]
    yx <- cancor(cbind(f(y),1),cbind(f(x),1))$cor[1]
  }else{
    xy <- dcor(cbind(f(x),1),cbind(f(y),1))
    yx <- dcor(cbind(f(y),1),cbind(f(x),1))
  }

  return(max(xy,yx))
}


testRDC <- function(data, ohe, featureTypes, linear) {
  adjm <- matrix(0, ncol=ncol(data), nrow = ncol(data))

  inputpos <- t(combn(ncol(data),2))

  columnnames <- names(data)

  rdccoef <- foreach(i = 1:nrow(inputpos), .combine=rbind) %dorng% {
    c1 <- inputpos[i,1]
    c2 <- inputpos[i,2]

    d1 <- data[,c1]
    d2 <- data[,c2]

    if(ohe){

      if(featureTypes[[c1]] == "discrete" || featureTypes[[c1]] == "binary"){
        d1 <- dummy(columnnames[c1], data, sep="_")
      }


      if(featureTypes[[c1]] == "discrete" || featureTypes[[c1]] == "binary"){
        d2 <- dummy(columnnames[c2], data, sep="_")
      }
    }

    rdcv1 <- rdc(d1,d2, linear=linear)


    return(rdcv1)
  }
  for (i in 1:nrow(inputpos)){
    adjm[inputpos[i,1], inputpos[i,2]] = rdccoef[i]
  }
  diag(adjm) <- 1
  #write.table(adjm,"/tmp/adj.txt",sep=",",row.names=FALSE)
  return(adjm)
}
