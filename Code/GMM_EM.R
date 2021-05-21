
#####################################################
# Teaching implementation of EM for mixture models 
# Author: Mattias Villani, http://mattiasvillani.com
#####################################################

mixtureEM <- function(data, K, initMu, initSigma, initPi, tol){
  
  # Preliminaries
  count <- 0
  nTotal <- length(data)
  N <- rep(0,K)
  gamma = matrix(0,nTotal,K)
  Mu = initMu
  Sigma = initSigma
  Pi = initPi
  
  LogLOld <- 10^10
  LogLDiff <- 10^10
  while (LogLDiff > tol){
    count <- count + 1
    # E-step
    
    for (k in 1:K){
      gamma[,k] = Pi[k]*dnorm(data, mean=Mu[k], sd = sqrt(Sigma[k]))
    }
    sumGamma <- rowSums(gamma)
    for (k in 1:K){
      gamma[,k] = gamma[,k]/sumGamma
    }
    
    # M-step
    for (k in 1:K){
      N[k] <- sum(gamma[,k])
      Mu[k] = (1/N[k])*sum(gamma[,k]*data)
      Sigma[k] = sqrt((1/N[k])*sum(gamma[,k]*(data-Mu[k])^2))
      Pi[k] = N[k]/nTotal
    }
    
    # Log-Likelihood computation
    for (k in 1:K){
      gamma[,k] = Pi[k]*dnorm(data, mean=Mu[k], sd = sqrt(Sigma[k]))
    }
    LogL = sum(log(rowSums(gamma)))
    print(c(LogL,LogLOld))
    LogLDiff = abs(LogL - LogLOld)
    LogLOld = LogL
    
  }
  return(list(Mu = Mu,Sigma = Sigma, Pi = Pi, LogL = LogL, nIter = count))
}

# Generate some data for testing
data <- c(rnorm(50,1,1), rnorm(50,5,0.5))
initMu = c(-1,1)
initSigma = c(1,1)
initPi = c(0.5,0.5)

# Run the EM
K <- 2
EMfit <- mixtureEM(data, K, initMu, initSigma, initPi, tol = 0.0000001)
EMfit

# Plot the histogram and density fit
dataGrid <- seq(min(data)-0.2*IQR(data),max(data)+0.2*IQR(data), length = 1000)
densEst <- rep(0,length(dataGrid))
for (k in 1:K){
  densEst <- densEst + EMfit$Pi[k]*dnorm(dataGrid, mean=EMfit$Mu[k], sd = sqrt(EMfit$Sigma[k]))
}
hist(data, 30, freq = FALSE, xlim = c(min(data)-0.2*IQR(data),max(data)+0.2*IQR(data)))
lines(dataGrid,densEst, col="red", lwd = 2)


# Analyze the fish data
fish <- read.table('https://github.com/mattiasvillani/MLcourse/raw/main/Data/Fish.dat')
data <- as.matrix(fish)

# Run the EM
# Plot the histogram and density fit
dataGrid <- seq(min(data)-0.2*IQR(data),max(data)+0.2*IQR(data), length = 1000)
par(mfrow = c(2,2))
for (K in 1:4){
  densEst <- rep(0,length(dataGrid))
  initMu <- quantile(data,seq(1/(K+1),1-1/(K+1), length=K)) + 10*rnorm(K)
  EMfit <- mixtureEM(data, K, initMu = initMu, initSigma  = rep(sd(data),K), initPi = rep(1/K,K), tol = 0.0000001)
  for (k in 1:K){
    densEst <- densEst + EMfit$Pi[k]*dnorm(dataGrid, mean=EMfit$Mu[k], sd = sqrt(EMfit$Sigma[k]))
  }
  hist(data, 30, freq = FALSE, ylim = c(0,0.15), xlim = c(min(data)-0.2*IQR(data),max(data)+0.2*IQR(data)))
  lines(dataGrid,densEst, col="red", lwd = 2)
}


