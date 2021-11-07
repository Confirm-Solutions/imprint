library(devtools)
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

bckt_tune(n_sim=n_sim, 
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
