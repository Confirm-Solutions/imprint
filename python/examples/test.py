import pykevlar.core as core
import numpy as np
import pickle
from multiprocessing import Pool

def f(gr_pkl, i):
    gr_upkl = pickle.loads(gr_pkl)
    print(f"Thread {i}")

def g(gr):
    gr_pkl = pickle.dumps(gr)

    with Pool(16) as p:
        p.starmap(f, [(gr_pkl, i) for i in range(16)])

np.random.seed(321)

d = 3
n = 10000
gr = core.GridRange(d, n)
gr.thetas()[...] = np.random.normal(size=(d, n))
gr.radii()[...] = 0.5

null_hypos = [core.HyperPlane(np.ones(d), 0)]

gr.create_tiles(null_hypos)
print(gr.n_tiles())
gr.prune()
print(gr.n_tiles())
print(gr.thetas())

g(gr)
