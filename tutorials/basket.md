# An introduction to analyzing trial designs with Imprint.



We're going to analyze the Type I Error a three arm basket trial following the design of [Berry et al. (2013)](https://pubmed.ncbi.nlm.nih.gov/23983156/).

Critically, the log-odds for each arm of the trial are assumed to be drawn from a shared normal distribution. This hierarchical design leads to a sharing effect between the log-odds for the different arms.

\begin{align}
\mathbf{y} &\sim \mathrm{Binomial}( \mathbf{p}, \mathbf{n})\\
\mathbf{p} &= \mathrm{expit}(\mathbf{\theta} + logit(\mathbf{p_1}))\\
\mathbf{\theta} &\sim N(\mu, \sigma^2)\\
\mu &\sim N(\mu_0, S^2)\\
\sigma^2 &\sim \mathrm{InvGamma}(0.0005, 0.000005)\\
\end{align}



## Part 0: Type I Error

First, we'll show off how easy it is to use `imprint`. If none of this makes sense, jump to the next section where we'll get into details. The next cell:
1. Constructs a grid of tiles. (inside `ip.cartesian_grid`)
2. Simulates the Basket trial model (inside `ip.validate`)
3. Constructs a 99% confidence upper bound on the Type I Error of the Basket trial for each tile. (inside `ip.validate`)

```python
from scipy.special import logit
import matplotlib.pyplot as plt
import numpy as np

import imprint as ip
from imprint.models.basket import BayesianBasket, FastINLA
```

```python
g = ip.cartesian_grid(
    # The minimum and maximum values of each parameter.
    theta_min=[-3.5, -3.5, -3.5],
    theta_max=[1.0, 1.0, 1.0],
    # The number of tiles in each dimension.
    n=[16, 16, 16],
    # Define the null hypotheses.
    null_hypos=[ip.hypo(f"theta{i} < {logit(0.1)}") for i in range(3)],
)
validation_df = ip.validate(
    BayesianBasket,
    g=g,
    # The threshold for our rejection criterion.
    lam=0.05,
    # The number of simulations to perform for each tile.
    K=2000,
    # This is the binomial n parameter, the number of patients recruited to each arm of the trial.
    model_kwargs={"n_arm_samples": 35},
)
```

```python
ip.setup_nb()
plt.figure(figsize=(10, 4), constrained_layout=True)
theta_tiles = g.get_theta()
t2 = np.unique(theta_tiles[:, 2])[4]
selection = theta_tiles[:, 2] == t2

plt.subplot(1, 2, 1)
plt.title(f"slice: $\\theta_2 \\approx$ {t2:.1f}")
cntf = plt.tricontourf(
    theta_tiles[selection, 0],
    theta_tiles[selection, 1],
    validation_df["tie_est"][selection],
)
plt.tricontour(
    theta_tiles[selection, 0],
    theta_tiles[selection, 1],
    validation_df["tie_est"][selection],
    colors="k",
    linestyles="-",
    linewidths=0.5,
)
cbar = plt.colorbar(cntf)
cbar.set_label("Simulated fraction of Type I errors")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.axis("square")

plt.subplot(1, 2, 2)
cntf = plt.tricontourf(
    theta_tiles[selection, 0],
    theta_tiles[selection, 1],
    validation_df["tie_bound"][selection],
)
plt.tricontour(
    theta_tiles[selection, 0],
    theta_tiles[selection, 1],
    validation_df["tie_bound"][selection],
    colors="k",
    linestyles="-",
    linewidths=0.5,
)
cbar = plt.colorbar(cntf)
cbar.set_label("Bound on the fraction of Type I errors")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.axis("square")

plt.show()
```

## Part 1: Constructing a parameter grid

The first step to constructing an upper bound on Type I Error is to chop up the domain into "tiles". Each of these tiles will be a hyperrectangle with a simulation point at the center.

We're going to use the `ip.cartesian_grid` function to produce a 3 dimensional set of points covering $\theta_i \in [-3.5, 1.0]$. The points lie at the center of (hyper)rectangular cells. The cells cover the whole box.

The resulting imprint object is a `Grid`.

```python
g_raw = ip.cartesian_grid(
    theta_min=[-3.5, -3.5, -3.5], theta_max=[1.0, 1.0, 1.0], n=[16, 16, 16]
)
type(g_raw)
```

Most of the information contained in the `Grid` object is contained in a Pandas DataFrame with details on each tile.

```python
g_raw.df.head()
```

We can access some of the important data in this raw "database" data format with convenience functions:
- `Grid.get_theta()` - return an NxD array containing the centers of each tile.
- `Grid.get_radii()` - return an NxD array containing the half-width (aka, "radius") of each tile.

```python
g_raw.get_theta()[:5]
```

```python
g_raw.get_radii()[:5]
```

Since we want to discuss Type I Error, we need a null hypothesis to discuss! There are handy tools in imprint for defining a null hypothesis. From a computational perspective, most null hypotheses have a bounding hyperplane that defines the sharp null. But, defining a plane by hand is sort of a pain, so we have tools for translating a symbolic statement into such a bounding plane for a null hypothesis space. For example:

```python
ip.hypo("theta0 > theta1 - 0.1")
```

Our null hypothesis here will be that $\theta_i < \mathrm{logit}(0.1)$ for $i = 0, 1, 2$.

```python
logit(0.1)
```

```python
null_hypos = [
    ip.hypo(f"theta0 < -2.1972"),
    ip.hypo(f"theta1 < -2.1972"),
    ip.hypo(f"theta2 < -2.1972"),
]
```

Once we have defined these planes, we attach the null hypothesis to the grid created above using `Grid.add_null_hypos`. For each hyperrectangular cell, the method intersects with the null hypothesis boundaries and splits into multiple tiles whenever a cell is intersected by a null hypothesis plane.

```python
g_unpruned = g_raw.add_null_hypos(null_hypos)
```

We can see that the tiles now have `null_truth` columns. Each of these columns represents whether that particular null hypothesis is true or false on that tile.

```python
g_unpruned.active().df.head(n=10)
```

Next, for the sake of investigating Type I Error, we only care about regions of space where the null hypothesis is true! 

In order to reduce computational effort, we can "prune" our grid by removing any tiles that are entirely in the alternative hypothesis space for all hypotheses.

```python
g = g_unpruned.prune()
```

```python
g_unpruned.n_tiles, g.n_tiles
```

All our previous steps can be condensed into a single call to `cartesian_grid`:

```python
g = ip.cartesian_grid(
    theta_min=[-3.5, -3.5, -3.5],
    theta_max=[1.0, 1.0, 1.0],
    n=[16, 16, 16],
    null_hypos=[ip.hypo(f"theta{i} < {logit(0.1)}") for i in range(3)],
    prune=True,
)
```

**At this point, you can skip to the next section if you're not interested in learning about the details of the grid object.**

Here, we'll grab a few of the important variables from the grid object and examine them. First, let's look at the center of each tile in the grid. The shape of the array will be `(n_tiles, 3)` because we have 3 parameter values per point.


```python
theta_tiles = g.get_theta()
theta_tiles.shape
```

```python
unique_t2 = np.unique(theta_tiles[:, 2])
unique_t2
```

In the figure below, we plot $\theta_0$ and $\theta_1$ for a couple different values of $\theta_2$. You can see that the shape of the domain in $(\theta_0, \theta_1)$ changes depending on whether $\theta_2$ is in the null space for arm 2 or not. The solid white region without any tile centers in the right figure represents the region where the alternative hypothesis is true for all three arms. The solid black lines represent the boundaries of the arm 0 and the arm 1 null hypothesis boundary planes.


```python
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.title(f"$\\theta_2 = {unique_t2[3]}$")
selection = theta_tiles[:, 2] == unique_t2[3]
plt.plot(theta_tiles[selection, 0], theta_tiles[selection, 1], "k.")
plt.hlines(logit(0.1), -4, 2)
plt.vlines(logit(0.1), -4, 2)
plt.axis("square")
plt.xlim(np.min(theta_tiles[:, 0]) - 0.2, np.max(theta_tiles[:, 0]) + 0.2)
plt.ylim(np.min(theta_tiles[:, 1]) - 0.2, np.max(theta_tiles[:, 1]) + 0.2)

plt.subplot(1, 2, 2)
plt.title(f"$\\theta_2 = {unique_t2[10]}$")
selection = theta_tiles[:, 2] == unique_t2[10]
plt.plot(theta_tiles[selection, 0], theta_tiles[selection, 1], "k.")
plt.hlines(logit(0.1), -4, 2)
plt.vlines(logit(0.1), -4, 2)
plt.axis("square")
plt.xlim(np.min(theta_tiles[:, 0]) - 0.2, np.max(theta_tiles[:, 0]) + 0.2)
plt.ylim(np.min(theta_tiles[:, 1]) - 0.2, np.max(theta_tiles[:, 1]) + 0.2)
plt.show()
```

Let's explore another useful array produced for the grid. `g.get_null_truth()` will contain whether the null hypothesis is true for each arm for each tile. Because we have three arms and three null hypotheses, this array has the same shape as `theta_tiles`. In a different trial where, for example, we are comparing two arms, the number of null hypotheses may be different than the number of parameters.


```python
g.get_null_truth().shape
```

Since we've pruned the grid, the tiles are all in the null hypothesis space for at least one arm.


```python
np.all(np.any(g.get_null_truth(), axis=1))
```

## Part 2: Simulating to compute type I error rates and gradients



Now that we've constructed and examined our computation grid, let's actually compute type I error and its gradient.

First, in order to do this, we need to build an inference algorithm that tells us whether to reject or not given a particular dataset. We're going to use an implementation of INLA applied to the model described at the beginning of this section. INLA is a method for approximate Bayesian inference. The `fi.test_inference` function below will implement this inference algorithm and and return a test statistic. The details of this inference are not particularly important to what we're doing here so we'll leave it unexplained. 

First, we'll check that the inference does something reasonable. Assuming a threshold of 0.05, it rejects the null for arms 1 and 2 where the success counts are 5 and 9 but does not reject the null for arm 0 where the success count is 4. This seems reasonable!


```python
y = [[4, 5, 9]]
n = [[35, 35, 35]]
fi = FastINLA(n_arms=3, critical_value=0.95)
fi.test_inference(np.stack((y, n), axis=-1))
```

```python
import jax
import jax.numpy as jnp


class BayesianBasket:
    def __init__(self, seed, K):
        self.n_arm_samples = 35

        # The family field is used by imprint to determine how to compute the Tilt Bound.
        self.family = "binomial"

        # A binomial family needs to know N!
        self.family_params = {"n": self.n_arm_samples}

        # Everything below here is internal to the model and is not needed by
        # imprint.
        np.random.seed(seed)
        self.samples = np.random.uniform(size=(K, self.n_arm_samples, 3))
        self.fi = FastINLA(n_arms=3)

    def sim_batch(self, begin_sim, end_sim, theta, null_truth, detailed=False):
        # 1. Calculate the binomial count data.
        # The sufficient statistic for binomial is just the number of uniform draws
        # above the threshold probability. But the `p_tiles` array has shape (n_tiles,
        # n_arms). So, we add empty dimensions to broadcast and then sum across
        # n_arm_samples to produce an output `y` array of shape: (n_tiles,
        # sim_size, n_arms)
        p = jax.scipy.special.expit(theta)
        y = jnp.sum(self.samples[None] < p[:, None, None], axis=2)

        # 2. Determine if we rejected each simulated sample.
        # fi.test_inference expects inputs of shape (n, n_arms) so we must flatten
        # our 3D arrays. We reshape exceedance afterwards to bring it back to 3D
        # (n_tiles, sim_size, n_arms)
        y_flat = y.reshape((-1, 3))
        n_flat = jnp.full_like(y_flat, self.n_arm_samples)
        data = jnp.stack((y_flat, n_flat), axis=-1)
        test_stat_per_arm = self.fi.test_inference(data).reshape(y.shape)

        return jnp.min(
            jnp.where(null_truth[:, None, :], test_stat_per_arm, jnp.inf), axis=-1
        )
```

```python
sims = BayesianBasket(0, 100).sim_batch(0, 100, theta_tiles, g.get_null_truth())
```

```python
sims.shape
```

```python
rejections = sims < 0.05
n_rejections = np.sum(rejections, axis=1)
```

```python
plt.figure(figsize=(5, 4), constrained_layout=True)
select = theta_tiles[:, 2] == np.unique(theta_tiles[:, 2])[4]
plt.scatter(
    theta_tiles[select, 0], theta_tiles[select, 1], c=n_rejections[select], s=50
)
cbar = plt.colorbar()
cbar.set_label(r"Number of sims with p-value $<$ 0.05")
plt.title(f"slice: $\\theta_2 \\approx$ {t2:.1f}")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.axis("square")
plt.show()
```

Next, the bound calculation will be done by `ip.validate`. This function will:
1. Simulate $K = 2000$ times for every tile.
2. Decide to reject based on the test statistic: $T < \lambda$ where $\lambda = 0.05$.
3. Compute a Clopper-Pearson confidence interval on the Type I Error at the simulation points.
4. Compute a Tilt-Bound-based confidence interval on the Type I Error over each tile.


```python
%%time
validation_df = ip.validate(BayesianBasket, g=g, lam=0.05, K=2000)
```

Looking at the results, we see four columns:
- `tie_sum`: The raw number of rejections for each tile.
- `tie_est`: The Monte Carlo estimate of Type I Error at the simulation point.
- `tie_cp_bound`: The Clopper-Pearson upper confidence bound on the Type I Error at the simulation point.
- `tie_bound`: The Tilt-Bound upper confidence bound on the Type I Error over the whole tile.

```python
validation_df.head()
```

Before continuing, let's look at a couple slices of the type I error estimates:


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4), constrained_layout=True)
for i, t2_idx in enumerate([4, 8]):
    t2 = np.unique(theta_tiles[:, 2])[t2_idx]
    selection = theta_tiles[:, 2] == t2

    plt.subplot(1, 2, i + 1)
    plt.title(f"slice: $\\theta_2 \\approx$ {t2:.1f}")
    plt.scatter(
        theta_tiles[selection, 0],
        theta_tiles[selection, 1],
        c=validation_df["tie_est"][selection],
        s=90,
    )
    cbar = plt.colorbar()
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    cbar.set_label("Simulated fraction of Type I errors")
plt.show()
```

Note that the upper bound here is going to be quite loose because we have a very coarse grid. The looseness of the bound will be quadratic in cell size because of the second order term. In addition, there is a lot of error in our pointwise type I error estimate because the number of simulations is only 2000.



## Step 4: 3D Bound visualization



For this last step, we're going to visualize the bound with a Plotly 3D visualization tool.

```python
bound_components = np.array(
    [
        validation_df["tie_est"],
        validation_df["tie_cp_bound"] - validation_df["tie_est"],
        validation_df["tie_bound"] - validation_df["tie_cp_bound"],
        validation_df["tie_bound"],
    ]
).T
t2 = np.unique(theta_tiles[:, 2])[4]
selection = theta_tiles[:, 2] == t2

np.savetxt("P_tutorial.csv", theta_tiles[selection, :].T, fmt="%s", delimiter=",")
np.savetxt("B_tutorial.csv", bound_components[selection, :], fmt="%s", delimiter=",")
```

<!-- #region -->
Open [the frontend installation instructions](../../frontend/README.md) and follow them. Copied here:

1. On Mac: `brew install node`. Elsewhere, figure out how to install nodejs!
2. Install reactjs with `npm i react-scripts`

Finally:

```bash
cd frontend
npm start
```

Click on "Upload B matrix" and choose the B matrix we just saved. Do the same for the P matrix. Now you should be able to play around with the 3D visualization! Also, you can select the different layers to see the magnitude of different bound components.

<!-- #endregion -->
