#helper functions
logit = function(p){ log(p) - log(1-p)}
invlogit = function(theta){1/(1 + exp(-theta))}

#note: V0 should generally be equal to the prior sigma-inverse
d=4
dirtymultilevel = function(n = rep(100,d), phat = rep(.5,d) ,mu0 = rep(0,d),V0 = diag(rep(1,length(phat)))){
  MLE = logit(phat)
  sample_I= diag(n*phat*(1-phat))
  precision_posterior = V0 + sample_I
  Sigma_posterior = solve(precision_posterior)
  mu_posterior = Sigma_posterior %*% (sample_I %*% MLE + V0 %*% mu0)
  #implement 95% CI on each arm
  CI_upper_logit = mu_posterior + 1.96*sqrt(diag(Sigma_posterior))
  median_logit = mu_posterior
  CI_lower_logit = mu_posterior - 1.96*sqrt(diag(Sigma_posterior))
  conf_logit = cbind( CI_lower_logit, median_logit, CI_upper_logit)
  colnames(conf_logit) = c("lower","median","upper")
  conf_prob = invlogit(conf_logit)
  return(list( mu_posterior = mu_posterior, Sigma_posterior = Sigma_posterior, conf_logit = conf_logit, conf_prob = conf_prob) )
}

#This one is intended to match Ben's example!
d=4
dirtymultilevel(n = rep(50,d), phat = c(28,14,33,36)/50,mu0 = rep(0,d),V0 = diag(rep(1,d)))
# the agreement isn't perfect, but I think not bad!

# Output:
# $mu_posterior
#            [,1]
# [1,]  0.2230568
# [2,] -0.8592214
# [3,]  0.6090148
# [4,]  0.8592214

# $Sigma_posterior
#            [,1]       [,2]       [,3]       [,4]
# [1,] 0.07507508 0.00000000 0.00000000 0.00000000
# [2,] 0.00000000 0.09025271 0.00000000 0.00000000
# [3,] 0.00000000 0.00000000 0.08183306 0.00000000
# [4,] 0.00000000 0.00000000 0.00000000 0.09025271

# $conf_logit
#            lower     median      upper
# [1,] -0.31397989  0.2230568  0.7600935
# [2,] -1.44804632 -0.8592214 -0.2703965
# [3,]  0.04832785  0.6090148  1.1697018
# [4,]  0.27039646  0.8592214  1.4480463

# $conf_prob
#          lower    median     upper
# [1,] 0.4221436 0.5555341 0.6813740
# [2,] 0.1903024 0.2975020 0.4328098
# [3,] 0.5120796 0.6477160 0.7630911
# [4,] 0.5671902 0.7024980 0.8096976
