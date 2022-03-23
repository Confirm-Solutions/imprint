# make sure the input parameters are the same as ParallelBinomialSampling.R!
#A# Requires a clean set of saved simulation outputs in the folder: tempdir
#A# Be aware that TotalK is different here from the sub-simulations K! 

simslist = list.files("tempdir")
M = length(simslist)
for(m in 1:M){
  filepath = paste("tempdir",simslist[m],sep = "/")
  load(filepath)
  
  if(m == 1){
    TypeIest_Combined = matrix(NA, nrow = length(TypeIest), ncol = M)
    TypeI_squaredev_Combined = matrix(NA, nrow = length(TypeIest), ncol = M)
    Gradest_Combined = array(NA, c(dim(Gradest), M))
  }
  
  TypeIest_Combined[,m] = TypeIest
  TypeI_squaredev_Combined[,m] = TypeI_squaredev
  Gradest_Combined[,,m] = Gradest
}

TypeIest = apply(TypeIest_Combined,1,mean)
TypeIsquaredev = apply(TypeI_squaredev_Combined,1,mean)
TypeI_standarderror = sqrt(TypeI_squaredev_Combined) #N# The denominator is n here, not n-1
Gradest = apply(Gradest_Combined,c(1,2),mean)

Kequals =  strsplit(simslist[m],split = c("\\."))[[1]][3]
K = as.integer(sub("..","",Kequals,))
TotalK = K*M

#####
###############
#PARAMETER DETAILS HERE!
###############


{d = 2
p_threshold = 0.6
posterior_threshold = 0.95
# Max number of patients in the trial:
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
 # source("ImportanceSampler.R")
#install.packages("plot3D") 
# library(plot3D)
}

#Simulation Parameters
{#K = 10 #10000
 # gradmethod = "cantelli" #or "normal"
 # epsilon = 1/2^4 #1/(2^7?)
 # delta = 0.01
 # Thetaj = gridEps(epsilon)
 # J = nrow(Thetaj)
 # p = t(apply(Thetaj,1, unlink)) 
  #set.seed(12375)
}

load("ParametersList.RData")
#Contents:
#save(K,epsilon, delta, gradmethod, Thetaj, J, p, file ="ParametersList.RData")

#########Below is copied from BinomialMonteCarlo

ucbf = TypeIest + (1/sqrt(TotalK))*TypeI_standarderror*qnorm(p = 1 - (delta/2),) #R#
# Could replace the normal interval with a more conservative interval, if desired

#Here is the second-order bound:
h = rep(epsilon,d)
remainderBound = secondOrderBoundBinomial(Taumax=Taumax, p_j = p, h = h)

# Next, computations for the Grad f confidence bound:
# Now comes the sup-f argument
if(gradmethod == "cantelli"){
  alpha = delta/2
  lambdasquared = (1/TotalK)* ((1/alpha) - 1) * secondOrderBoundBinomial(Taumax = Taumax,
                                                                    p_j = p, h = h)
  ucbgradf = apply(Gradest,1, function(z){sum( abs(z * h))}) + sqrt(lambdasquared)
}

upperBoundThetaj = ucbf + ucbgradf + remainderBound

save(upperBoundThetaj,ucbf,ucbgradf,remainderBound, TypeIest,Gradest, TypeI_standarderror, 
     file = "Binomial_Combined.RData")
#Now make and save some graphs!

#library(ggplot2)
#plotname = paste("Binomial_TotalK",TotalK,"eps",epsilon,".pdf",sep = "")
#pdf(file = plotname)
#qplot(Thetaj[,1],Thetaj[,2],size = upperBoundThetaj) + scale_size_area()
#dev.off()
