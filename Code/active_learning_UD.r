#############################################################
### Machine Learning, Stockholm University, VT2021        ###
### Active learning, illustrating example                 ###
### Frank Miller, 2021-05-24                              ###
#############################################################

# choose query strategy for active learning: uncertainty sampling ("U") or D-optimality ("D")
querys <- "D"
# size of unlabeled data pool
n      <- 100 
# first part until init uses random sampling
init   <- 15  
# final number to be labeled
end    <- 45

# generate 2-dimensional GMM x-data with two components N((0, 2), diag(1, 1)) and N((2, 2), diag(1, 1))
ragr  <- rbinom(n, size=1, prob=0.5)  # help-variable
data  <- cbind(rnorm(n, mean=2*ragr, sd=1), rnorm(n, mean=2, sd=1))
# simulation of true class (y-data, group variable not known to the learner)
group <- rbinom(n, size=1, prob=1/(1+exp(3-3*data[,1])))

# label randomly for initialisation (lab variabel known to the learner)
lab    <- rep(NA, n)
rindex <- sample(n, init)
lab[rindex] <- group[rindex]

#Plot with unlabeled data and initial random labeling
plot(data, col=1, xlab=expression(x[1]), ylab=expression(x[2]))
points(data, col=4-2*lab, lwd=2, pch=16)

# initialize vector for saving accuracy development
accdev <- NULL
# X0 is design matrix for all x-data; first a column with 1's for the intercept, then the two features
X0 <- cbind(rep(1, n), data)

# sequential labeling
for (i in (init+1):end){ 
  # Maximum Likelihood Estimate in logistic regression 
  lgm   <- glm(lab ~ data[, 1] + data[, 2], family="binomial")
  beta  <- summary(lgm)$coef[, 1]

  # compute predictions for all data points
  predic  <- 1/(1+exp(-beta[1]-beta[2]*data[,1]-beta[3]*data[,2]))
  # compute accuracy
  accura  <- 1-mean(abs((predic>0.5)-group))
  accdev  <- cbind(accdev, c(i-1, accura))
  if (querys=="U"){
    uncert  <- abs(predic-0.5)
    uncert[!is.na(lab)] <- NA
    # determine index of data point to be queried
    index   <- which.min(uncert)
  }
  if (querys=="D"){
    # XL is design matrix for labeled data; VL is diagonal of W-matrix for labeled data
    XL   <- X0[!is.na(lab), ]
    G0   <- 1/(1+exp(-X0 %*% beta)) 
    V0   <- G0 * (1-G0)
    VL   <- V0[!is.na(lab)]
    crit <- rep(NA, n)
    for (j in 1:n){
      if (is.na(lab[j])){
        # X is design matrix XL plus an additional unlabeled data point j; V accordingly
        X    <- rbind(XL, X0[j, ])
        V    <- c(VL, V0[j])
        # Information matrix if x_j is added and D-criterion
        Infj <- t(X) %*% diag(V) %*% X
        crit[j] <- 1/det(Infj)
      }
    }
    # determine index of data point to be queried
    index <- which.min(crit)
  }
  # oracle labels the queried data point
  lab[index] <- group[index]
  points(t(data[index, ]), col=4-2*lab[index], lwd=2, pch=19)
}


