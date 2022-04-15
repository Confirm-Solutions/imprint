# In this file, we are going to take the dirtyBayes algorithm from dirtymultilevel.R, and make it:

# (1) use Sherman-Morrison
# (2) parallelize over many different versions of the prior V0. Actually, screw this, let's just not do it!
# (3) perform an integration over the prior for sigma^2

# Google doc:
# https://docs.google.com/document/d/1_gGqdUV8eXTOSZTX8BMut5YKKnFIN5tIECyx0uG34z0/edit

fast_invert = function(S,d){
  for(k in 1:length(d)){
    offset = ( d[k]/(1 + d[k]*S[k,k])) * outer(S[k,], S[,k],"*") #I wonder how to cheaply represent this outer? but in C++ it should be just a trivial for-loop anyway
    S = S - offset
  }
  S
}
{#S_0 = diag(rep(1,4)) + .1
#S = S_0
#d = c(.1,.2,.3,.4)
#for(k in 1:length(d)){
#  offset = ( d[k]/(1 + d[k]*S[k,k])) * outer(S[k,], S[,k],"*") #I wonder how to cheaply represent this outer? but in C++ it should be just a trivial for-loop anyway
#  S = S - offset
#}
#testvalue = solve(solve(S_0) + diag(d))
#S - testvalue #this matches up to very close precision! awesome!
# So now we'll put it into a function:
# Note here that S is sigma, and d is the precision of the data likelihood, thus d is in the INVERSE space of Sigma

# What is not awesome, is that local changes to either the values of d or S can have effects throughout the whole sequence of computations
# So I don't see ANY gain yet from parallelization. Too bad!
}# Sherman Morrison Testing Section

# Integration preliminaries
require(statmod)
require(invgamma)
n <- 50
#let's evaluate the endpoints of the prior in logspace-sigma:
#determine endpoints:
a=log(1e-8)
b=log(1e3)
gleg <- gauss.quad(n, kind = "legendre")
pts = (gleg$nodes+ 1)*(b-a)/2 + a
wts = gleg$weights * (b-a)/2 #sum(wts) = b-a so it averages to 1 over space
density_logspace <-dinvgamma(exp(pts), shape = .0005, rate = .000005)*exp(pts) # the extra exp(pts) account for Jacobian
#plot(density_logspace)
#sum(density_logspace*wts) #this matches with
#pinvgamma(exp(3),shape = .0005, rate = .000005) - pinvgamma(exp(-8),shape = .0005, rate = .000005)

#helper functions
logit = function(p){ log(p) - log(1-p)}
invlogit = function(theta){1/(1 + exp(-theta))}
d=4
p_threshold = rep(.4,d)
#This function implements dirtyBayes on a multilevel model, given a known phat, and a fixed multilevel covariance matrix S_0 and precision V_0 = S_0^-1
fastndirty_thresh = function(thresh = logit(p_threshold),n = rep(100,d), phat = rep(.5,d) ,mu_0 = rep(0,d),
                              S_0 = diag(rep(1,length(phat))), V_0 = diag(rep(1,length(phat)))){
  MLE = logit(phat)
  sample_I= n*phat*(1-phat) #diag(n*phat*(1-phat))
  #precision_posterior = V0 + sample_I
  #Sigma_posterior = solve(precision_posterior) # This line and the line above have been replaced by the call below to fast_invert
  Sigma_posterior = fast_invert(S_0,sample_I)
  mu_posterior = Sigma_posterior %*% (sample_I * MLE + V_0 %*% mu_0)
  {#implement 95% CI on each arm
    #CI_upper_logit = mu_posterior + 1.96*sqrt(diag(Sigma_posterior))
    #median_logit = mu_posterior
    #CI_lower_logit = mu_posterior - 1.96*sqrt(diag(Sigma_posterior))
    #conf_logit = cbind( CI_lower_logit, median_logit, CI_upper_logit)
    #colnames(conf_logit) = c("lower","median","upper")
    #conf_prob = invlogit(conf_logit)
    #return(list( mu_posterior = mu_posterior, Sigma_posterior = Sigma_posterior, conf_logit = conf_logit, conf_prob = conf_prob) )
  } #confidence interval code
  
  #What we now must return instead, is posterior threshold exceedance probabilities for each arm.
  thresh_exceed = pnorm( (mu_posterior - thresh) / sqrt(diag(Sigma_posterior)))
  #return(list( thresh_exceed = thresh_exceed, mu_posterior = mu_posterior, Sigma_posterior = Sigma_posterior ))
  return(thresh_exceed)
}

{#Nice, this matches what we get from our slow function call!
#S_0 = diag(rep(1,d)) + .1
#V_0 = solve(S_0)
#fastndirty_thresh(n = rep(50,d), phat = c(28,14,33,36)/50,mu_0 = rep(0,d),V_0 = V_0,S_0 = S_0)
#dirtymultilevel_postthresh(n = rep(50,d), phat = c(28,14,33,36)/50,mu0 = rep(0,d),V0 = V_0)
# d=4
# dirtymultilevel_postthresh = function(thresh = rep(logit(.4),d),n = rep(100,d), phat = rep(.5,d) ,mu0 = rep(0,d),V0 = diag(rep(1,length(phat)))){
#   MLE = logit(phat)
#   sample_I= diag(n*phat*(1-phat))
#   precision_posterior = V0 + sample_I
#   #Sigma_posterior = solve(precision_posterior)
#   mu_posterior = Sigma_posterior %*% (sample_I %*% MLE + V0 %*% mu0)
#   {#implement 95% CI on each arm
#   #CI_upper_logit = mu_posterior + 1.96*sqrt(diag(Sigma_posterior))
#   #median_logit = mu_posterior
#   #CI_lower_logit = mu_posterior - 1.96*sqrt(diag(Sigma_posterior))
#   #conf_logit = cbind( CI_lower_logit, median_logit, CI_upper_logit)
#   #colnames(conf_logit) = c("lower","median","upper")
#   #conf_prob = invlogit(conf_logit)
#   #return(list( mu_posterior = mu_posterior, Sigma_posterior = Sigma_posterior, conf_logit = conf_logit, conf_prob = conf_prob) )
#   } #confidence interval code
#   
#   #What we now must return instead, is posterior threshold exceedance probabilities for each arm.
#   thresh_exceed = pnorm( (mu_posterior - thresh) / sqrt(diag(Sigma_posterior)))
#   #return(list( thresh_exceed = thresh_exceed, mu_posterior = mu_posterior, Sigma_posterior = Sigma_posterior ))
#   return(thresh_exceed)
# }
# dirtymultilevel_postthresh(n = rep(50,d), phat = c(28,14,33,36)/50,mu0 = rep(0,d),V0 = diag(rep(1,d)))
  } #Test code to check that fastndirty_thresh is giving us what we wanted!

#Now, let's integrate the fastndirty outputs over sigma-squared!
conditional_exceed_prob_given_sigma = function(sigma_sq, mu_sig_sq,mu_0 = rep(0,d),
                                               phat = c(28,14,33,36)/50, n=rep(50,d), thresh = logit(p_threshold)){ #there are lots of other implicit arguments as well.
  S_0 = diag(rep(sigma_sq,d)) + mu_sig_sq
  #V_0 = solve(S_0) #but because this is a known case of the form aI + bJ, we can use the explicit
  # inverse formula, given by: 1/a I - J*(b/(a(a+db)))
  V_0 = diag(rep(1/sigma_sq,d)) - (mu_sig_sq / sigma_sq) / (sigma_sq + d *mu_sig_sq) #Note, by the way, that it's probably possible to use significant precomputation here
  print(V_0)
  print(S_0)
  print(thresh)
  print(phat)
  fastndirty_thresh(thresh=thresh, n = n, phat = phat,mu_0 = mu_0,V_0 = V_0,S_0 = S_0)
}
#test checks out, giving familiar answer:
#conditional_exceed_prob_given_sigma(sigma_sq = 1, mu_sig_sq = .1)

#This reweighting is taken from pages 14-16 from INLA approaches, and implements equation (*)
posterior_reweight = function(sigma_sq, mu_sig_sq, mu_prior, phat, n){
  print("posterior_reweight")
  sample_I = n*phat*(1-phat)
  total_var = diag(rep(sigma_sq,d)) + mu_sig_sq + diag(1/sample_I)
  thetahat = logit(phat)
  print(thetahat)
  print(sample_I)
  print(total_var)
  det(total_var)^(-1/2) * exp( (-1/2)* sum( (t(thetahat - mu_prior) %*% solve(total_var))* (thetahat - mu_prior)) )
}#dropping the factor of 2pi

#Now, getting the conditional-probabilities of exceedance and putting them into the integration loop:
# conditional_exceed_probs <-sapply(exp(pts), conditional_exceed_prob_given_sigma, mu_sig_sq = 10)
# posterior_sigma_reweights <- sapply(exp(pts), posterior_reweight, mu_sig_sq = 10, mu_prior = rep(0,d), phat = c(28,14,33,36)/50, n = rep(50,d))
# total_reweights = posterior_sigma_reweights * density_logspace * wts
# normalized_reweights = total_reweights/sum(total_reweights) #this now gives us an effective posterior density for sigma-squared
# posterior_exceedance_probs <- t(conditional_exceed_probs)*normalized_reweights #this automatically fills in density_logspace by column
# apply(posterior_exceedance_probs,2,sum)


##############



###########3
#Trying one of the Berry replications - we'll take the second interim analysis
# Data: n_i = 15. [3, 8, 5, 4] are the results
n = rep(15,d)
phat = c(3,8,5,4)/15
p_threshold = rep(.3,4)

conditional_exceed_prob_given_sigma(exp(pts[30]), n= n, phat = phat,mu_sig_sq = 100, thresh = logit(p_threshold))
##########
conditional_exceed_probs <-sapply(exp(pts), conditional_exceed_prob_given_sigma, n= n, phat = phat,
                                  mu_sig_sq = 100, thresh = logit(p_threshold))
posterior_sigma_reweights <- sapply(exp(pts), posterior_reweight, mu_sig_sq = 100, mu_prior = rep(-1.34,d), phat = phat, n = n)
total_reweights = posterior_sigma_reweights * density_logspace * wts
normalized_reweights = total_reweights/sum(total_reweights) #this now gives us an effective posterior density for sigma-squared
print("norm_reweights")
print(normalized_reweights)
posterior_exceedance_probs <- t(conditional_exceed_probs)*normalized_reweights #this automatically fills in density_logspace by column
p_threshold
apply(posterior_exceedance_probs,2,sum)



####################### SHOVING MASS NEAR sigma^2 = ZERO INTO A POINT AT ZERO

#Noting that likelihoods are essentially identical when sigma-squared < 10^-6, 
#we can save computation by just driving this into a point mass at zero.
# Sigma = 0 literally just means that all of the arms are identical
# The problem totally collapses into 1-d Bayes, with just a normal prior on logit(p):

# phat_collapsed = mean(phat)
# n_collapsed = d*50
# collapsed_sig_sq = mu_sig_sq
# collapsed_mu_0 = 0 #this doesn't need any change

# #now we are just doing a 1-dimensional dirtyBayes
# collapsed_thresh = function(thresh = logit(p_threshold),#note that thresh is still multi-dimensional
#                             phat_collapsed = mean(phat), n_collapsed = 200,
#                             collapsed_mu_0 = 0,collapsed_sig_sq = 1){
#   MLE = logit(phat_collapsed)
#   sample_I= n*phat_collapsed*(1-phat_collapsed) #diag(n*phat*(1-phat))
#   #precision_posterior = V0 + sample_I
#   #Sigma_posterior = solve(precision_posterior) # This line and the line above have been replaced by the call below to fast_invert
#   Sigma_posterior = 1/(1/collapsed_sig_sq + sample_I) #fast_invert(S_0,sample_I)
#   mu_posterior = Sigma_posterior %*% (sample_I * MLE + (1/collapsed_sig_sq) * collapsed_mu_0)
#   {#implement 95% CI on each arm
#     #CI_upper_logit = mu_posterior + 1.96*sqrt(diag(Sigma_posterior))
#     #median_logit = mu_posterior
#     #CI_lower_logit = mu_posterior - 1.96*sqrt(diag(Sigma_posterior))
#     #conf_logit = cbind( CI_lower_logit, median_logit, CI_upper_logit)
#     #colnames(conf_logit) = c("lower","median","upper")
#     #conf_prob = invlogit(conf_logit)
#     #return(list( mu_posterior = mu_posterior, Sigma_posterior = Sigma_posterior, conf_logit = conf_logit, conf_prob = conf_prob) )
#   } #confidence interval code
  
#   #What we now must return instead, is posterior threshold exceedance probabilities for each arm.
#   thresh_exceed = pnorm( (c(mu_posterior) - thresh) / sqrt(Sigma_posterior))
#   #return(list( thresh_exceed = thresh_exceed, mu_posterior = mu_posterior, Sigma_posterior = Sigma_posterior ))
#   return(thresh_exceed)
# }
