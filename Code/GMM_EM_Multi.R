
#########################################################################
# Teaching implementation of EM for multivariate Gaussian mixture models 
# Author: Mattias Villani, http://mattiasvillani.com
#########################################################################
library(mvtnorm)

mixtureMultiGaussianEM <- function(data, M, initMu, initSigma, initPi, tol){
  
  # data is a n x p matrix with n observations on p variables.
  # initMu is an p x M matrix with initial values for the component means
  # initSigma is an p x p x M 3D array with initial values for the component covariance matrices
  # initPi is a M-dim vector with initial values for the component probabilities
  
  # Preliminaries
  count <- 0
  n <- dim(data)[1]
  nHat <- rep(0,M)
  W = matrix(0,n,M)  # n x m matrix with weights for all observations and all components.
  Mu = initMu        
  Sigma = initSigma
  Pi = initPi
  unitVect = rep(1,n) # Just a vector of ones that we need later for efficiency
  
  LogLOld <- 10^10
  LogLDiff <- 10^10
  while (LogLDiff > tol){
    count <- count + 1
    
    # E-step
    
    for (m in 1:M){
      W[,m] = Pi[m]*dmvnorm(data, Mu[,m], Sigma[,,m])
    }
    sum_w <- rowSums(W)
    for (m in 1:M){
      W[,m] = W[,m]/sum_w
    }
    
    # M-step
    for (m in 1:M){
      nHat[m] <- sum(W[,m])
      Mu[,m] = (1/nHat[m])*crossprod(W[,m],data)
      Res = data - tcrossprod(unitVect,Mu[,m])
      Sigma[,,m] = (1/nHat[m])*t(Res)%*%diag(W[,m])%*%Res # Matrix version of the estimate in the slides
      Pi[m] = nHat[m]/n
    }
    
    # Log-Likelihood computation - to check convergence
    for (m in 1:M){
      W[,m] = Pi[m]*dmvnorm(data, Mu[,m], Sigma[,,m])
    }
    LogL = sum(log(rowSums(W)))
    LogLDiff = abs(LogL - LogLOld)
    LogLOld = LogL
    
  }
  return(list(Mu = Mu, Sigma = Sigma, Pi = Pi, LogL = LogL, nIter = count))
}

# Generate some data for testing
n = 100
data <- rbind(rmvnorm(n/2, c(0,0), 0.2*diag(2)), rmvnorm(n/2, c(1,1), 1*diag(2)))
initMu = matrix(rnorm(2*2),2,2)
initSigma = array(NA,c(2,2,2))
initSigma[,,1] = 1*diag(2)
initSigma[,,2] = 1*diag(2)
initPi = c(0.5,0.5)

# Run the EM
M <- 2
EMfit <- mixtureMultiGaussianEM(data, M, initMu, initSigma, initPi, tol = 0.0000001)

# Compare with mixtools package - remember that the order of the components are arbitrary
# Initial values matter so we don't always get the same results, but sometimes, 
# so implementation seems correct.

#library(mixtools)
#mixtoolsFit = mvnormalmixEM(data, k = M)

#EMfit$Mu
#mixtoolsFit$mu

#EMfit$Sigma
#mixtoolsFit$sigma

#EMfit$Pi
# mixtoolsFit$lambda
