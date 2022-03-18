library(devtools)
library(plot3D)
load_all()

n_sim = 100000
alpha = 0.025
delta = 0.025
ph2_size = 50
n_samples = 250
grid_dim = 3
lower = -0.5
upper = 0.5
p_size = 64

radius = grid_radius(p_size, lower, upper)
p = make_grid(p_size, lower, upper)
p = 1/(1 + exp(-p))
p_endpt = make_endpts(p_size, lower, upper)
p_endpt = 1/(1+exp(-p_endpt))
thr_vec = make_grid(20, 1.95, 3.)
thr_vec = sort(thr_vec, decreasing=T)

f_output_name = "bckt_out"

#tune.out = bckt_tune(
#          n_sim=n_sim, 
#          alpha=alpha, 
#          delta=delta, 
#          ph2_size=ph2_size, 
#          n_samples=n_samples, 
#          grid_dim=grid_dim, 
#          grid_radius=radius,
#          p=p, 
#          p_endpt=p_endpt, 
#          lmda_grid=thr_vec, 
#          start_seed=0)
#thr = tune.out$lmda
#print(tune.out$err)
#print(thr)

thr = 1.96
# if lmda is within the range, fit at lmda.
#if (thr_vec[length(thr_vec)] <= thr & thr <= thr_vec[1]) {
    bckt_fit(
        n_sim=n_sim,
        delta=delta,
        ph2_size=ph2_size,
        n_samples=n_samples,
        grid_dim=grid_dim,
        grid_radius=radius,
        p=p,
        p_endpt=p_endpt,
        lmda=thr,
        f_output_name,
        start_seed=69,
        p_batch_size=p_size**grid_dim
        )
#}

# unserialize the output and compute the upper bound
out = unserialize(f_output_name)
abs_grad = apply(out$grad, 1, function(x) sum(abs(x)))
upper_bd = out$c + out$c_bd + radius * abs_grad + out$grad_bd + out$hess_bd

upper_bd_arr <- array(upper_bd, dim=c(length(p), length(p), length(p)))
c_arr <- array(out$c, dim=c(length(p), length(p), length(p)))
c_bd <- array(out$c_bd, dim=c(length(p), length(p), length(p)))
g_arr <- radius * array(abs_grad, dim=c(length(p), length(p), length(p)))
g_bd <- array(out$grad_bd, dim=c(length(p), length(p), length(p)))
hess_bd <- array(out$hess_bd, dim=c(length(p), length(p), length(p)))

# 3d-plot
plot.new()
persp3D(x = p,y = p,z=upper_bd_arr[,,length(p)/2], 
        ticktype = "detailed", xlab = "p-1", ylab = "p-2", zlab = "Type I Error", 
        cex.axes = .5,
        main = paste("Type I Error Upper Bound (delta = ", as.character(delta),")"))
filename = paste("Binomial3D_",3,'_',length(p),'.png',sep="")
png(filename)
dev.off()

