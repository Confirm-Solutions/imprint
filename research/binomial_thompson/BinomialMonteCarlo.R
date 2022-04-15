### Task List:

# Get Parallel Binomial and Gaussian simulations running on Sherlock
# Derive On Paper How to Compute a Gradient for a Smooth Design Feature
# Fool around with & Possibly Implement Massively Constrained SGD

##############
#BACKSTORY

# Let's next code up a real example: binomial Thompson sampling in two dimensions.

# This example will emulate an uncontrolled two-arm phase II trial.
# We will use a total of N = 100 samples, split adaptively.

# Let's imagine we have two arms, and we wish to do a Bayesian
# selection of just one of them (or neither.)

# In order to maximize our Bayesian power:

# Let's declare the rejection rule to be:
# -Do not reject the arm with smaller posterior probability of p_i > .6
# -Do reject the arm with greater posterior probability of p_i > .6, if that posterior
#   probability is greater than c = 70% [note this is a fairly arbitrary choice; c is tunable.]

# Note: in general these methods can absolutely optimize for other metrics besides posterior power.
# In general, given a reward function f(x, p) [above is the case f(p) = 1_{p >.6}],
# we can do arm selection according to max-above-some-cutoff of E[f(x, p)].
# In a you-can-only-pick-one-and-some-of-the-time decision problem [i.e. Type I error constrained], 
# given a fixed trial stopping rule, I believe this approach to be optimal.
#####################

###############
#TRIAL DETAILS
###############
{d = 2
  p_threshold = 0.6
  posterior_threshold = 0.95
  # The process for simulating our trial then proceeds according to a loop:
  Nmax = 100
  # Prior distributions
  # priors = Beta(alpha = 1, beta = 1) #this is uniform, flat prior on p
  d = 2
  alphas = 1 #alphas = rep(1,d) #R# NOT ready for general priors or higher dimen
  betas = 1 #betas = rep(1,d) #R#
  ##################
  # Parameter information:
  ##################
  # We will work in the natural parameter space:
  # That is, Theta = log(p/1-p)
  # Our null hypothesis boundary will be p = 0.6. This corresponds to Theta = 0.4054651
  link = function(p){
    log(p/(1-p))
  }
  unlink = function(theta){
    exp(theta)/(1 + exp(theta))
  }
  
  lowerLimitofTheta = -.5
  upperLimitoftheta = 1.5
  upperNullLimit = link(p_threshold)
  Taumax = rep(Nmax,d)
}


#SOURCE FILES
{source("ParameterSpace.R")
  source("SimulateData.R")
  source("LikelihoodModel.R")
  source("Taylor.R")
  #source("ImportanceSampler.R")
  library(testthat)
  library(parallel)
  library(doParallel)
  library(plot3D)
}

#Simulation Parameters
{K = 10 #10000
  gradmethod = "cantelli" #or "normal"
  epsilon = 1/2^2 #1/(2^7)
  delta = 0.01
  Thetaj = gridEps(epsilon)
  J = nrow(Thetaj)
  p = t(apply(Thetaj,1, unlink)) # Moving back into real space
  
  #set.seed(12375)
}

#This example code does one trial simulation:
#trial_output<- simulateThomp(p = c(.5,.5), alphaprior = alphas,betaprior = betas,totalN = Nmax)
#BetaReject(trial_output = trial_output,p_threshold = p_threshold,posterior_threshold = posterior_threshold)

####################################
# Re-write all of the code below!
TypeIest = rep(NA,J)
Gradest = matrix(NA,nrow =J , ncol = d)
ucbf = rep(NA,J)
ucbgradf = rep(NA,J)
remainderBound = rep(NA,J)

#nullstatus = inNull(Theta)
#  if(any(nullstatus)){
#Simulate Data

#######################################################
#RESTART HERE! WE ARE REBUILDING THE SIMULATION FUNCTION
#######################################################
    rejections = array(NA, dim = c(K, J, d)) #lapply(rep(NA,J),matrix,nrow = K, ncol = d)
    gradients = array(NA, dim = c(K, J, d)) #lapply(rep(NA,J),matrix,nrow = K, ncol = d)
    TypeI = matrix(NA, nrow= K,ncol = J)#lapply(rep(NA,J),rep,times= K)
    
    ptm <- proc.time()
    for(k in 1:K){
      print(paste(k," of ",J))
      x<-proc.time() - ptm
      print(x)
      print("Estimated Minutes Remaining:")
      print(x[3] * (K -k) / k / 60)
      #previously: # trial_output<- simulateThomp(p = unlink(Theta), alphaprior = alphas,betaprior = betas,totalN = Nmax)
      trial_output <- simulateThompFast(p = p,alphaprior = alphas, betaprior =betas, Nmax = 100)
      
      rejections[k,,] = BetaRejectFast(post_params = trial_output$posterior_params,
                                       p_threshold = p_threshold,posterior_threshold = posterior_threshold)[[1]]
      gradients[k,,] = BinomialGradFast(trial_output,p)
    }
   
    
    nullstatus = array(NA, dim = c(K, J, d))
    temp = inNull(Thetaj)
    for(k in 1:K){
      nullstatus[k,,] = temp
    }
    falserejs = rejections & nullstatus
    typeIerror = apply(falserejs,c(1,2),any)
    TypeIest = apply(typeIerror, 2, mean)
    TypeI_standarderror = apply(typeIerror, 2, sd)
    ucbf = TypeIest + (1/sqrt(K))*TypeI_standarderror*qnorm(p = 1 - (delta/2),) #R#
    # Could replace the normal interval with a more conservative interval, if desired
    
    #Here is the second-order bound:
    h = rep(epsilon,d)
    remainderBound = secondOrderBoundBinomial(Taumax=Taumax, p_j = p, h = h)
    
    # Next, computations for the Grad f confidence bound:
    typeIerrord = array(data = typeIerror ,dim = c(K, J, d))
    gradreject = typeIerrord * gradients
    Gradest = apply(gradreject, c(2,3), mean) #Mean estimate of the gradient
    # Now comes the sup-f argument
    # Note: with cantelli, there is no scaling with K going on. 
    # So it's OK that this isn't super fast.
    if(gradmethod == "cantelli"){
      alpha = delta/2
      lambdasquared = (1/K)* ((1/alpha) - 1) * secondOrderBoundBinomial(Taumax = Taumax,
                                                                       p_j = p, h = h)
      for(j in 1:J){
        #Would this be faster to re-write using apply? I wonder...
      ucbgradf[j] = sum(abs(Gradest[j,]*h)) + sqrt(lambdasquared[j])
      }
    }
    #      covbound = diag(Taumax*p[j,]*(1-p[j,]))
    # h = rep(epsilon,d)
    # V = expand.grid(rep(list(c(-1, 1)), d))
    # cardh = nrow(V)
    # linHolder = rep(NA,cardh)
    # alph = delta/2
    # 
    #  c = cov.wt(gradreject[,j,]) #R# rewrite if I want this at all!
    #if(gradmethod == "normal"){
    #   for(i in 1:cardh){
    #     hprime = V[i,]*h
    #     ucbLinearized = sum(Gradest[j,]*hprime) + 
    #       (1/sqrt(K))*qnorm(1-delta/2)*sqrt( t(t(hprime)) %*% c$cov %*% t(hprime))
    #     linHolder[i] = ucbLinearized
    #   }
    # }



save.image(file = "BinomFastExample.RData")

#Continue graphing from here!
upperBoundThetaj = ucbf + ucbgradf + remainderBound
library(ggplot2)
qplot(Thetaj[,1],Thetaj[,2],size = upperBoundThetaj) + scale_size_area()
qplot(Thetaj[,1],Thetaj[,2],size = TypeIest) + scale_size_area()
qplot(Thetaj[,1],Thetaj[,2],size = ucbf - TypeIest) + scale_size_area()
qplot(Thetaj[,1],Thetaj[,2],size = ucbgradf) + scale_size_area()
qplot(Thetaj[,1],Thetaj[,2],size = upperBoundThetaj - TypeIest) + scale_size_area() #Total slack



#############
# 3-D plot
############

par(mar=c(1,1,1,5))
persp3D(x = unique(Thetaj[,1]),y = unique(Thetaj[,2]),z=matrix(upperBoundThetaj,nrow = sqrt(J),ncol= sqrt(J)), 
        ticktype = "detailed", xlab = "theta-1", ylab = "theta-2", zlab = "Type I Error", 
        cex.axes = .5,
        main = paste("Type I Error Upper Bound (delta = ", as.character(delta),")"))

filename = paste("Binomial3D",K,J,"pdf",sep=".")

pdf(file = filename)
persp3D(x = unique(Thetaj[,1]),y = unique(Thetaj[,2]),z=matrix(upperBoundThetaj,nrow = sqrt(J),ncol= sqrt(J)), 
        ticktype = "detailed", xlab = "theta-1", ylab = "theta-2", zlab = "Type I Error", 
        cex.axes = .5,
        main = paste("Type I Error Upper Bound (delta = ", as.character(delta),")"))
dev.off()
