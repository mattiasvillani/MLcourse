# Analysing the Lidar data using Splines
# Author: Mattias Villani, https://mattiasvillani.com

library(SemiPar) # to get the lidar data
library(splines) # for natural cubic splines, ns(), and B-splines, bs().
library("RColorBrewer") # for pretty colors
colors = brewer.pal(12, "Paired")[c(1,2,7,8,3,4,5,6,9,10)];

# Read data and plot
data(lidar)
lidar$range = (lidar$range-mean(lidar$range))/sd(lidar$range)
plot(lidar$range, lidar$logratio, pch = 19, cex = 0.5, ylim = c(-1,0.15), xlim = c(-1.65,1.65),
     main = "lidar data", xlab = "range (standardized)", ylab = "logratio")

# linear model
model = lm(logratio ~ range, data = lidar)
xGrid = seq(-3,3,length = 1000)
yPred = predict(model, newdata = data.frame(range = xGrid))
lines(xGrid, yPred, col = colors[2], lwd = 2)

# 5th degree polynomial model
model = lm(logratio ~ poly(range, 5), data = lidar)
xGrid = seq(-3,3,length = 1000)
yPred = predict(model, newdata = data.frame(range = xGrid))
lines(xGrid, yPred, col = colors[4], lwd = 2)

# Natural cubic spline with 5 knots
model = lm(logratio ~ ns(range, knots = seq(-1.5,1.5, length = 5)), data = lidar)
xGrid = seq(-3,3,length = 1000)
yPred = predict(model, newdata = data.frame(range = xGrid))
lines(xGrid, yPred, col = colors[6], lwd = 2)

# Natural cubic spline with 10 knots
model = lm(logratio ~ ns(range, knots = seq(-1.5,1.5, length = 10)), data = lidar)
xGrid = seq(-3,3,length = 1000)
yPred = predict(model, newdata = data.frame(range = xGrid))
lines(xGrid, yPred, col = colors[10], lwd = 2)

legend(x = "topright", inset=.05, legend = c("Linear", "Poly order 5","Spline 5 knots", "Spline 10 knots"),  
       lty = c(1, 1, 1, 1), lwd = c(2, 2, 2, 2), pch = c(NA,NA,NA,NA),
       col = c(colors[2], colors[4], colors[6], colors[10]))

mydata = data.frame(y = rnorm(100), x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))

lm(y ~ ns(x1, df = 10) + ns(x3, df = 10), data = mydata)
