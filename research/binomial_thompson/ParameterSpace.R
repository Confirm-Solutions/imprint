#Details of Theta for a simple Gaussian Example
inNull = function(Theta){
  Theta <= upperNullLimit
}

#Assumes a square frame, 
# and a scalar epsilon (rather than vector) that divides the null hyp space evenly
gridEps = function(epsilon, d = 2){
  s = seq(lowerLimitofTheta+epsilon,upperLimitoftheta-epsilon,by = 2*epsilon)
  as.matrix(expand.grid(rep(list(s), d)))
}

#Draws randomly from the square, bounded by lowerLimitofTheta and upperLimitofTheta
# in each coordiante
sampleTheta = function(d){
  lowerLimitofTheta + (upperLimitoftheta-lowerLimitofTheta)*runif(d)
}

falseReject = function(Theta,H){
  any(inNull(Theta) & H)
}
