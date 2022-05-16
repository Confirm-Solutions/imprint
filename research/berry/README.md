# List of notebooks for the Berry work

**Work products**: intended to be useful and somewhat understandable.

- intro_to_inla.ipynb:
  - building inla from scratch, implementing the method entirely in this notebook. demonstrating some of the ideas and features.
  - comparing gaussian arm marginals to laplace approximation arm marginals.
  - also, building dirty bayes from scratch.
- berry_part1.ipynb:
  - explanation of dirty bayes.
  - figures comparing DB vs INLA vs MCMC vs QUAD on Berry Figure 1/2.
  - earlier version of the arm marginal figures. the newer figures are in berry_marginal_playground. i think I could pull something useful from these figures.
- berry_part3_simulation.ipynb
  - reproduce the later figures from the berry paper where they simulate for various sets of known parameter values and determine the type I error rate.
- berry_kevlar.ipynb:
  - running the berry kevlar model in 2 and 4d via kevlar and using the python accumulator tools.
  - c++ berry-inla implementation runner.
  - building a rejection table.
  - comparing against mcmc to make sure the rejection inference is doing well.
  - running parallel accumulation using both jax and multiprocessing.
- berry_marginal_playground.ipynb:
  - Tools for making inla, mcmc and quadrature plots of 1) arm marginals, 2) arm densities given sigma^2, 3) hyperparameter posteriors.

**Notebooks mostly just for development**: somewhat more inscrutable.

- quadrature_dev.ipynb:
  - exploring the domain of integration.
  - developing the eigenvector approach.
  - tools for exploring a slice of the quadrature grid.
