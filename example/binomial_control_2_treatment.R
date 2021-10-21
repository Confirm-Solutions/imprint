source("ParameterSpace.R")

# @param    p       probability vector for arms 1-3
# @param    u       uniform(0,1) rv as n x 3 matrix where 
#                   column j is for arm j.
# @param    n.ph2   number of patients (the first n.ph2 rows of u)
#                   to be considered in phase II.
bc2t.simulate <- function(p, u, n.ph2) {

    # Phase II

    a2 <- u[1:n.ph2,2] < p[2]
    a3 <- u[1:n.ph2,3] < p[3]

    arm_star <- (sum(a3) > sum(a2)) + 2
    a_star <- u[,arm_star] < p[arm_star]
    a1 <- u[,1] < p[1]

    # Phase III
    
    n <- dim(u)[1]
    p_star <- mean(a_star)
    p_1 <- mean(a1)
    var_star <- p_star * (1-p_star) / n
    var_1 <- p_1 * (1-p_1) / n
    z <- (p_star - p_1) / sqrt(var_star + var_1)
    z
}

unlink <- function(theta) {
    exp(theta)/(1 + exp(theta))
}

n.ph2 <- 50
n <- 250
n.arms <- 3
epsilon <- 1/2^2
Thetaj <- gridEps(epsilon, d=n.arms)
p <- t(apply(Thetaj, 1, unlink)) # Moving back into real space
u <- matrix(runif(n * n.arms), ncol=n.arms)
z <- apply(p, 1, function(prow) bc2t.simulate(prow,u,n.ph2))
print(z)
