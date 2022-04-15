### A Parallelized Version of BinomialMonteCarlo

# Intended to generate simulations in parallel, with a SLURM cluster.
# Another file will collate the relevant outputs, which will be partially analyzed
# to shrink them in size
slurm_arrayid <- Sys.getenv('SLURM_ARRAY_TASK_ID')
print("ID:")
print(slurm_arrayid)

###############
#TRIAL DETAILS
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
  #source("ImportanceSampler.R")
  
}

#Simulation Parameters
{K = 1000 #10000
  gradmethod = "cantelli" #or "normal"
  epsilon = 1/2^7 #1/(2^7?)
  delta = 0.01
  Thetaj = gridEps(epsilon)
  J = nrow(Thetaj)
  #set.seed(12375)
}

#This example code does one trial simulation:
#trial_output<- simulateThomp(p = c(.5,.5), alphaprior = alphas,betaprior = betas,totalN = totN)
#BetaReject(trial_output = trial_output,p_threshold = p_threshold,posterior_threshold = posterior_threshold)

####################################
# Re-write all of the code below!

p = t(apply(Thetaj,1, unlink)) # Moving back into real space

if(slurm_arrayid == 1){
save(K,epsilon, delta, gradmethod, Thetaj, J, p, file ="ParametersList.RData")
}


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
  #previously: # trial_output<- simulateThomp(p = unlink(Theta), alphaprior = alphas,betaprior = betas,totalN = totN)
  trial_output <- simulateThompFast(p = p,alphaprior = alphas, betaprior =betas, Nmax = Nmax)
  
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
sqerr = function(z){  mean((z - mean(z))^2) }
#TypeI_standarderror = apply(typeIerror, 2, sd)
TypeI_squaredev = apply(typeIerror, 2, sqerr)
typeIerrord = array(data = typeIerror ,dim = c(K, J, d))
gradreject = typeIerrord * gradients
Gradest = apply(gradreject, c(2,3), mean) 


savedestination = paste(getwd(),"/tempdir/Binomial.ID=",slurm_arrayid,
                        ".K=",K, ".eps=",epsilon,sep  ="")

save(TypeIest, TypeI_squaredev, Gradest, file = savedestination)
