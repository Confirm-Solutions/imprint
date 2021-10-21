#Details of Theta for a simple Gaussian Example
inNull = function(Theta, upper){
  Theta <= upper
}

#Assumes a square frame, 
# and a scalar epsilon (rather than vector) that divides the null hyp space evenly
gridEps = function(epsilon, d = 2, lower = -0.5, upper = 1.5){
  s = seq(lower+epsilon,upper-epsilon,by = 2*epsilon)
  as.matrix(expand.grid(rep(list(s), d)))
}

#Draws randomly from the square, bounded by lowerLimitofTheta and upperLimitofTheta
# in each coordiante
sampleTheta = function(d, lower=-0.5, upper=1.5){
  lower + (upper-lower)*runif(d)
}

falseReject = function(Theta,H){
  any(inNull(Theta) & H)
}
