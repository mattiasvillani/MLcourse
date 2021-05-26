
#####################################################
# Teaching implementation of EM for mixture models 
# Author: Mattias Villani, http://mattiasvillani.com
#####################################################

mixtureEM <- function(data, M, initMu, initSigma, initPi, tol){
  
  # Preliminaries
  count <- 0
  n <- length(data)
  nHat <- rep(0,M)
  W = matrix(0,n,M)  # n x m matrix with weights for all observations and all components.
  Mu = initMu
  Sigma = initSigma
  Pi = initPi
  
  LogLOld <- 10^10
  LogLDiff <- 10^10
  while (LogLDiff > tol){
    count <- count + 1
    
    # E-step
    
    for (m in 1:M){
      W[,m] = Pi[m]*dnorm(data, mean=Mu[m], sd = Sigma[m])
    }
    sum_w <- rowSums(W)
    for (m in 1:M){
      W[,m] = W[,m]/sum_w
    }
    
    # M-step
    for (m in 1:M){
      nHat[m] <- sum(W[,m])
      Mu[m] = (1/nHat[m])*sum(W[,m]*data)
      Sigma[m] = sqrt((1/nHat[m])*sum(W[,m]*(data-Mu[m])^2))
      Pi[m] = nHat[m]/n
    }
    
    # Log-Likelihood computation - to check convergence
    for (m in 1:M){
      W[,m] = Pi[m]*dnorm(data, mean=Mu[m], sd = Sigma[m])
    }
    LogL = sum(log(rowSums(W)))
    LogLDiff = abs(LogL - LogLOld)
    LogLOld = LogL
    
  }
  return(list(Mu = Mu, Sigma = Sigma, Pi = Pi, LogL = LogL, nIter = count))
}

# Analyze the fish data
fish <- read.table('https://github.com/mattiasvillani/MLcourse/raw/main/Data/Fish.dat')
data <- as.matrix(fish)

# Run the EM and plot the histogram and density fit over a suitable grid
dataGrid <- seq(min(data)-0.2*IQR(data),max(data)+0.2*IQR(data), length = 1000)
par(mfrow = c(2,2))
for (M in 1:4){
  densEst <- rep(0,length(dataGrid))
  initMu <- quantile(data,seq(1/(M+1),1-1/(M+1), length=M)) + 10*rnorm(M)
  EMfit <- mixtureEM(data, M, initMu = initMu, initSigma  = rep(sd(data),M), initPi = rep(1/M,M), tol = 1e-7)
  for (m in 1:M){
    densEst <- densEst + EMfit$Pi[m]*dnorm(dataGrid, mean=EMfit$Mu[m], sd = EMfit$Sigma[m])
  }
  hist(data, 40, freq = FALSE, ylim = c(0,0.08), xlim = c(min(data)-0.2*IQR(data),max(data)+0.2*IQR(data)))
  lines(dataGrid,densEst, col="red", lwd = 2)
}


