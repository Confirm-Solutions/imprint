library(devtools)
library(plot3D)
load_all()

n_sim = 100000
alpha = 0.025
delta = 0.025
n_samples = 250
lower = -0.1/4
upper = 1/4
log_hzrd_lower = -(upper-lower)
log_hzrd_upper = 0.0
lmda_size = 32
censor_time = 2.0

radius = grid_radius(lmda_size, lower, upper)
lmda = exp(make_grid(lmda_size, lower, upper))
lmda_lower = exp(make_endpts(lmda_size, lower, upper)[1,])
hzrd_rate = exp(make_grid(lmda_size, log_hzrd_lower, log_hzrd_upper))
hzrd_rate_lower = exp(make_endpts(lmda_size, lower, upper)[1,])
thr_vec = make_grid(20, 1.95, 3.0)
thr_vec = sort(thr_vec, decreasing=T)

f_output_name = "eckt_out"

tune.out = eckt_tune(
          n_sim=n_sim, 
          alpha=alpha, 
          delta=delta, 
          n_samples=n_samples, 
          grid_radius=radius,
          censor_time=censor_time,
          lmda=lmda,
          lmda_lower=lmda_lower,
          hzrd_rate=hzrd_rate,
          hzrd_rate_lower=hzrd_rate_lower,
          thr_vec=thr_vec, 
          start_seed=0)
thr = tune.out$lmda
if (tune.out$err != "") print(tune.out$err)
print(thr)

# if lmda is within the range, fit at lmda.
if (tune.out$err == "") {
    eckt_fit(
        n_sim=n_sim,
        delta=delta,
        n_samples=n_samples,
        grid_radius=radius,
        censor_time=censor_time,
        lmda=lmda,
        lmda_lower=lmda_lower,
        hzrd_rate=hzrd_rate,
        hzrd_rate_lower=hzrd_rate_lower,
        thr=thr,
        f_output_name,
        start_seed=0)
}

# unserialize the output and compute the upper bound
out = unserialize(f_output_name)
abs_grad = apply(out$grad, 1, function(x) sum(abs(x)))
upper_bd = out$c + out$c_bd + radius * abs_grad + out$grad_bd + out$hess_bd

upper_bd_arr = matrix(upper_bd, nrow=lmda_size)
c_arr = matrix(out$c, nrow=lmda_size)

# 3d-plot
plot.new()
persp3D(x=lmda,y=hzrd_rate,z=c_arr, 
        ticktype = "detailed", xlab = "lmda", ylab = "hzrd_rate", zlab = "Type I Error", 
        cex.axes = .5,
        main = paste("Type I Error Upper Bound (delta = ", as.character(delta),")"))
filename = paste("Exp3D_",2,'_',lmda_size,'.png',sep="")
png(filename)
dev.off()
