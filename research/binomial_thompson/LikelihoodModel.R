#Details of the Likelihood Model for a Gaussian
# (or a stopped Gaussian?)

#X = c(10,5)
#sigma2 = c(1,2)
#Tau = c(20,10)
#Thetaj = c(.5,-.5)
#Theta0 = c(0,0)

RadonNikodym = function(Theta0,Thetaj,Xdata,Tau,sigma2){
  exp( - sum((1/2)*(Xdata - Thetaj*Tau )^2 / (sigma2*Tau))
       + sum((1/2)*(Xdata - Theta0*Tau )^2 / (sigma2*Tau))
         )
}

gradRadonNikodym = function(Theta0,Thetaj,Xdata,Tau,sigma2){
  (Xdata/sigma2 - Thetaj*Tau/sigma2)*RadonNikodym(Theta0,Thetaj,Xdata,Tau,sigma2)
}

secondOrderBound = function(Tau,sigma2, Theta = "NOT CODED YET"){
  diag(Tau/sigma2)
}

#######################

BinomialGradFast = function(trial_output,p){
  gradient = trial_output$successes - trial_output$n_arm*t(p)
  return(gradient)
}

##### Gradient for Binomial likelihood, without change of measure to a different Theta
BinomialGradMonte = function(trial_output,Theta){
  n_k = trial_output$successes + trial_output$failures
  X_k = trial_output$successes
  gradient = X_k - n_k *exp(Theta)/(1+exp(Theta))
}
#####

secondOrderBoundBinomial = function(Taumax=Taumax,p_j = "put a matrix here",h = c(1,1)){
  # Previously:
  #return(diag(Taumax * .25))
  
  # Now let's do it properly:
  apply(p, 1, function(z){
    sum(h^2 * Taumax*z*(1 - z))})
}
