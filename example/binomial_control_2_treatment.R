library(devtools)
library(plot3D)
load_all()

n_sim = 500
alpha = 0.05
delta = 0.05
ph2_size = 50
n_samples = 250
grid_dim = 3
lower = -0.5
upper = 1.5
p_size = 64

radius = grid_radius(n_samples, lower, upper)
p = make_grid(p_size, lower, upper)
p = 1/(1 + exp(-p))
p_endpt = make_endpts(p_size, lower, upper)
p_endpt = 1/(1+exp(-p_endpt))
thr_vec = make_grid(100, 12., 15.2)
thr_vec = sort(thr_vec, decreasing=T)

f_output_name = "bckt_out"

thr = bckt_tune(
          n_sim=n_sim, 
          alpha=alpha, 
          delta=delta, 
          ph2_size=ph2_size, 
          n_samples=n_samples, 
          grid_dim=grid_dim, 
          grid_radius=radius,
          p=p, 
          p_endpt=p_endpt, 
          lmda_grid=thr_vec, 
          start_seed=0)

# if lmda is within the range, fit at lmda.
if (12 < thr & thr < 15.2) {
    bckt_fit(
        n_sim=n_sim,
        alpha=alpha,
        delta=delta,
        ph2_size=ph2_size,
        n_samples=n_samples,
        grid_dim=grid_dim,
        grid_radius=radius,
        p=p,
        p_endpt=p_endpt,
        lmda=thr,
        f_output_name,
        start_seed=0)
}

# unserialize the output and compute the upper bound
out = bckt_unserialize(f_output_name)
abs_grad = apply(out$grad, 1, function(x) sum(abs(x)))
upper_bd = out$c + out$c_bd + radius * abs_grad + out$grad_bd + out$hess_bd

# 3d-plot
plot.new()
persp3D(x = p,y = p,z=matrix(upper_bd[1:length(p)^2],nrow=length(p)), 
        ticktype = "detailed", xlab = "p-1", ylab = "p-2", zlab = "Type I Error", 
        cex.axes = .5,
        main = paste("Type I Error Upper Bound (delta = ", as.character(delta),")"))
filename = paste("Binomial3D_",3,'_',length(p),'.png',sep="")
png(filename)
dev.off()

