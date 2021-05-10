
labelVar = "Sold"
plotVars = c("LogBook","MinBidShare")
condVars = c("PowerSeller","VerifyID","Sealed","MinBlem","LargNeg")
condVals = colMeans(df[,condVars])
df = training;
labels = unique(df[,labelVar])
idx = (df[,labelVar] == labels[1])
condVals
plot(df[idx,plotVars[1]], df[idx,plotVars[2]], col = colors[2], pch = 1, xlab = plotVars[1], ylab = plotVars[2])
points(df[idx==FALSE,plotVars[1]], df[idx==FALSE,plotVars[2]], col = colors[4], pch = 3)
legend(x = "topright", inset=.05, legend = c("Class 1", "Class 2"), pch = c(1, 3), col = c(colors[2], colors[4]))
#predict(glmnetFit$finalModel, newx = as.matrix(df[,c(plotVars,condVars)]))
#df[,c(plotVars,condVars)]


decisionplot <- function(model, df, labelVar, plotVars, condVars, condVals, colors) {
  
  # labelVar is a string with the name of the response variable with labels
  # plotVars is a vector with strings containing the names of the TWO variables in df used for plotting
  # condVars is a vector with strings containing the names of the remaining variables in used in model 
  # condVals is a numerical vector containing the conditioning values for variables in condVars
  unique(df[,labelVar])
  idx = (df[,labelVar] == labels[1])
  plot(df[idx,plotVars[1]], df[idx,plotVars[2]], col = colors[1], pch = 1)
  points(df[-idx,plotVars[1]], df[-idx,plotVars[2]], col = colors[2], pch = 2)
  
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}