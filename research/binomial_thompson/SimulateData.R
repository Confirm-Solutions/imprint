# Simulates a Gaussian trial with d dimensions
# We compute only the final sum statistics
# After Tau observations, each having variance sigma2
simulateTrial = function(d,Theta,Tau,sigma2){
  X = rnorm(d,mean =Theta*Tau,sd = sqrt(Tau*sigma2))
}

# Computes rejection based on a simple z-test
rejections = function(X,Tau,sigma2){
  est = X/Tau 
  lowbound = est - qnorm(.975)*sqrt(sigma2/Tau)
  lowbound >= 0
}

BayesRejections = function(X,Tau,sigma2,priormean, priorsigma2, positiveprob){
  #The sum of observations X_Tau has distribution
  # X ~ N(Tau*Theta,Tau*sigma2)
  # To put this onto the scale of the mean, we have
  # xbar = X/Tau ~ N(Theta,sigma2/Tau)
  # The prior is
  # Theta ~ N(priormean, priorsigma2)
  # This code will assume vector X of length > 1
  if(length(X) == 1){return("NEED MULTIDIMENSIONAL X")}
  xbar = X/Tau
  priorVinv = solve(priorsigma2)
  obsVinv = solve(diag(sigma2)/Tau) #Must divide by Tau because the observation here is the average xbar
  postV= solve(priorVinv + obsVinv) 
  postmu = postV %*% (priorVinv %*% priormean + obsVinv %*% xbar)
  #The posterior is then Theta ~ N(postmu, postV)
  # Next, we will reject entries of Theta where the posterior is positive with probability
  # equal to criticalprob
  sigs = diag(postV)
  zposts = postmu / sigs
  postprobs = pnorm(zposts)
  rejs = postprobs > positiveprob
  output =list(rejs,postprobs,zposts,postmu,postV)
  names(output) = unlist(strsplit("rejs,postprobs,zposts,postmu,postV",split =","))
  return(output)
}


# This function performs Thompson sampling. Currently being used by BinomialMonteCarlo.R
simulateThomp = function(p, alphaprior, betaprior, totalN){ #DEPRECATED BY THE FAST FUNCTION
  alphapost = alphaprior
  betapost = betaprior
  d = length(p)
  thomp_p = rep(NA,d)
  s_t = c(0,0)
  f_t = c(0,0)
  for(t in 1:totalN){
    #With Thompson probability, pick arm i
    for(i in 1:d){
      thomp_p[i] = rbeta(1,alphapost[i],betapost[i])
    }
    i_t = which(thomp_p == max(thomp_p))
    #Make draw from arm i with prob p[i].
    outcome = rbinom(1,1,p[i_t])
    #Tally success/failure (redundant, but good to check)
    s_t[i_t] = s_t[i_t] + outcome
    f_t[i_t] = f_t[i_t] + (1 - outcome)
    # Increment alpha or beta according to success or failure
    alphapost[i_t] = alphapost[i_t] + outcome
    betapost[i_t] = betapost[i_t] + (1 - outcome)
  }
  return(list(alphapost = alphapost, betapost = betapost,
              successes = s_t, failures= f_t))
}

#############WORKING HERE
# We are making the function: simulateThompFast

#It is dependent on:
posteriorDraw = function(Narm, success,a,b){
  #First: generate end1 ~ Gamma(alpha), end2 ~Gamma(beta)
  endleft = rgamma(n =1, shape = a)
  endright = rgamma(n =1, shape = b)
  # Then, generate N Gamma(1) R.V's: notate them g_m, for m in 1:n
  g = rgamma(n = max(Narm), shape = 1)
  # If the simulation has s successes, we let
  # X = end1 + Sum_{m = 1:s} g_m
  # X + Y = end2 + Sum_{m = 1:n} g_m
  gsums = c(0,cumsum(g))
  # Then, take all of these ratios of X/(X + Y); these are our draws from the posterior.
  betadraw = (gsums[success+1] + endleft)/ (gsums[Narm+1] + endleft + endright)
  betadraw
}

#Here is the meat:
simulateThompFast = function(p,alphaprior=1,betaprior = 1,Nmax = 100){ #R# alphaprior, betaprior assumed same across arms!
  if(!is.matrix(p)){return("Needs Matrix p. Currently accepts only ncol = 2")}
  d = ncol(p)
  J = nrow(p)
  a = alphaprior
  b = betaprior
  n0 = rep(0,J)
  n1 = rep(0,J)
  success0 = rep(0,J)
  success1 = rep(0,J)
  
for(npatient in 1:Nmax){
  #Doing step (1), posterior selection
  posterior0 = posteriorDraw(Narm = n0, success = success0, a = a, b = b)
  posterior1 = posteriorDraw(Narm = n1, success = success1, a = a, b = b)
  select_arm_1 = posterior1 > posterior0 #R# rewrite for d>2 arms
  
  #Now doing step (2), binomial drawing
  u = runif(1)
  outcome0 = rep(0, J)
  outcome1 = rep(0, J)
  outcome0 = outcome0 + ((select_arm_1 == 0) & p[,1] > u) #R# 
  outcome1 = outcome1 + ((select_arm_1 == 1) & p[,2] > u) #R#
  success0 = success0 + outcome0
  n0 = n0 + (select_arm_1 == 0) #R#
  success1 = success1 + outcome1
  n1 = n1 + (select_arm_1 == 1) #R#
}
  alphaposterior0 = success0 + a #R#
  alphaposterior1 = success1 + a #R#
  betaposterior0 = (n0 - success0) + b #R#
  betaposterior1 = (n1 - success1) + b #R#
  
  alphas = rbind(alphaposterior0, alphaposterior1)
  betas = rbind(betaposterior0,betaposterior1)
  posterior_params = list(alphas = alphas,betas = betas)
  
  successes = rbind(success0,success1)
  n_arm = rbind(n0,n1)
  
  output =list(successes = successes, n_arm = n_arm, 
          posterior_params = posterior_params)
  return(output)
}


BetaReject = function(trial_output,p_threshold=0.6,posterior_threshold = 0.7){ # DEPRECATED BY BETAREJECTFAST #R#
  d = length(trial_output$alphapost)
  postgreater = rep(NA,d)
  for(i in 1:d){
    postgreater[i] = 1 - pbeta(p_threshold, trial_output$alphapost[i], trial_output$betapost[i])
  }
  best_index = which(postgreater == max(postgreater))
  if(length(best_index)>1){
    best_index = sample(best_index,1)
  }
  rejs = rep(FALSE,d)
  if(postgreater[best_index] > posterior_threshold){
    rejs[best_index] = TRUE
  }
  return(list(rejections = rejs,posterior_exceedance_probs = postgreater))
}


### Likely Has a Bug!!

# Let's test the symmetry of BetaRejectFast:
BetaRejectFast = function(post_params,
                          p_threshold = p_threshold,posterior_threshold = 0.7){  
  alphamat = post_params$alphas
  betamat = post_params$betas
  d = nrow(alphamat)
  J = ncol(alphamat)
  postgreater = matrix(NA,nrow = d, ncol = J)
  for(arm in 1:d){
    for(j in 1:J){
      postgreater[arm,j] = 1 - pbeta(p_threshold, alphamat[arm,j], betamat[arm,j])
    }
  }
  best_index = max.col(t(postgreater))
   #We reject the arm if our Bayesian posterior for the arm is sufficiently large!
  rejs = matrix(FALSE,nrow = J,ncol = d)
  for(j in 1:J){
    if(postgreater[best_index[j],j] > posterior_threshold){
      rejs[j,best_index[j]] = TRUE
    }
  }
  return(list(rejections = rejs,posterior_exceedance_probs = postgreater))
}
