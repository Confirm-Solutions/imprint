import copy
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from itertools import product
from typing import Any
from typing import List

import numpy as np
import pandas as pd

import imprint.log
from .timer import unique_timer

logger = imprint.log.getLogger(__name__)


@dataclass
class NullHypothesis(ABC):
    def split(self, g: "Grid", curve_data: Any):
        """
        split returns a grid of new tiles that result from splitting the tiles
        in the input grid.

        This defaults to simply repeating each tile twice, but can be overridden
        to implement more sophisticated splitting strategies.

        Args:
            g: The input grid.
            curve_data: Arbitrary information describing the curve.
        """
        return g.repeat(2)

    @abstractmethod
    def dist(self, theta: np.ndarray):
        """
        Curve describes the signed distance of a point from the null hypothesis
        curve.

        For example, if the null hypothesis is a plane, curve should return
        `x.dot(n) - c`, where n is the normal vector and c is the offset.

        Args:
            theta: The points to evaluate the curve at. This array will be:
                (n_points, n_dims) shaped.

        Returns:
            distance: The signed distance of each point from the curve.
        """
        pass

    def use_fast_path(self):
        """
        Should we use the fast path or slow path?

        Fast path: The `dist` method is used as a first pass to determine which
        tile centers are within a single `radii` distance to the curve before
        using `side` to determine precisely which tiles intersect. This is much
        faster because it avoids expensive intersection tests in `side`. On the
        other hand, it can lead to incorrect results if the `dist` method does
        not accurately lower bound the true distance to the nearest point on
        the curve.

        Slow path: All tiles are passed to `side` to determine which tiles intersect.

        Returns:
            _description_
        """
        return False

    def side(self, g: "Grid"):
        """
        Determine which side of the null hypothesis curve each tile is on.
        If the tile is entirely above the curve, return 1. If the tile is
        entirely below the curve, return -1. If the tile intersects the curve,
        return 0.

        The default implementation checks whether the tile vertices are all
        above or below the curve. This is suitable for a variety of useful
        curves but will be incorrect for some more complicated curves.

        Args:
            g: The grid of tiles.

        Returns:
            intersects: an integer array for each tile, indicating whether:
                -1: the tile is entirely below the curve
                +1: the tile is entirely above the curve
                0: the tile intersects the curve
            curve_data: arbitrary information describing the curve that will be
                passed onwards to `split`.
        """
        _, vertices = g.get_theta_and_vertices()
        eps = 1e-15
        d = vertices.shape[-1]
        vertex_dist = self.dist(vertices.reshape((-1, d))).reshape(
            (-1, vertices.shape[1])
        )
        side = np.zeros(vertices.shape[0], dtype=np.int8)
        side[(vertex_dist >= -eps).all(axis=-1)] = 1
        side[(vertex_dist <= eps).all(axis=-1)] = -1
        return side, vertex_dist[side == 0]

    @abstractmethod
    def description(self):
        """
        A description of the null hypothesis so that it can be identified
        easily just by looking in the database.

        Returns:
            String description of the null hypothesis.
        """
        pass


@dataclass
class Grid:
    """
    A grid is a collection of tiles, each of which is a hyperrectangle in
    parameter space. The grid is stored as a pandas DataFrame, with one row per
    tile. The columns are:
    - id: A unique identifier for the tile. See gen_short_uuids for details on
      these ids.
    - active: Whether the tile is active. A tile is active if it has not been
      split.
    - parent_id: The id of the parent tile if the tile has been split. This is
      0 for tiles with no parent.
    - theta{i} and radii{i}: The center and half-width of the tile in the i-th
      dimension.

    Other columns may be added by other code. All columns will automatically be
    inherited in refinement and splitting operations.
    """

    df: pd.DataFrame
    worker_id: int
    null_hypos: List[NullHypothesis] = field(default_factory=lambda: [])

    @property
    def d(self):
        if not hasattr(self, "_d"):
            self._d = (
                max([int(c[5:]) for c in self.df.columns if c.startswith("theta")]) + 1
            )
        return self._d

    @property
    def n_tiles(self):
        return self.df.shape[0]

    @property
    def n_active_tiles(self):
        return self.df["active"].sum()

    def _add_null_hypo(self, H: NullHypothesis, inherit_cols: List[str]):
        hypo_idx = len(self.null_hypos)
        g_inactive = self.subset(~self.df["active"])
        g_inactive.df[f"null_truth{hypo_idx}"] = H.dist(g_inactive.get_theta()) >= 0

        g_active = self.prune_inactive()
        theta, vertices = g_active.get_theta_and_vertices()
        radii = g_active.get_radii()
        gridpt_dist = H.dist(theta)
        g_active.df[f"null_truth{hypo_idx}"] = gridpt_dist >= 0

        # If a tile is close to the curve, we need to check for intersection.
        # "close" is defined by whether the bounding ball of the tile
        # intersects the plane.

        # For each tile that is close to the plane, we ask the curve to
        # find which side of the plane the vertex lies on.
        if H.use_fast_path():
            close = np.ones(g_active.n_tiles, dtype=bool)
        else:
            close = np.abs(gridpt_dist) <= np.sqrt(np.sum(radii**2, axis=-1))
        side_close, curve_data = H.side(g_active.subset(close))
        side = np.zeros(g_active.n_tiles, dtype=np.int8)
        side[close] = side_close
        side[~close] = np.sign(gridpt_dist[~close])
        needs_split = side == 0

        # If H.dist just always returns 0, then we should update the null_truth
        # for all tiles here.
        g_active.df.loc[close, f"null_truth{hypo_idx}"] = side_close > 0

        if not needs_split.any():
            return g_inactive.concat(g_active)

        g_needs_split = g_active.subset(needs_split)
        g_split = H.split(g_needs_split, curve_data)

        # NOTE: currently assumes exactly two tiles are created for each split
        # tile.
        hypo_idx = len(g_needs_split.null_hypos)
        for i in range(hypo_idx):
            g_split.df[f"null_truth{i}"] = np.repeat(
                g_needs_split.df[f"null_truth{i}"].values, 2
            )
        null_truth = np.ones(g_split.n_tiles, dtype=bool)
        null_truth[1::2] = False
        g_split.df[f"null_truth{hypo_idx}"] = null_truth

        _inherit(g_split.df, g_needs_split.df, 2, inherit_cols)

        # Any tile that has been split should be ignored going forward.
        # We're done with these tiles!
        g_active.df["active"].values[needs_split] = False

        return g_inactive.concat(g_active, g_split)

    def add_null_hypos(
        self, null_hypos: List[NullHypothesis], inherit_cols: List[str] = []
    ):
        """
        Add null hypotheses to the grid. This will split any tiles that
        intersect the null hypotheses and assign the tiles to the null/alt
        hypothesis space depending on which side of the null hypothesis the
        tile lies. These assignments will be stored in the null_truth{i}
        columns in the tile dataframe.

        Args:
            null_hypos: The null hypotheses to add. List of NullHypothesis objects.
            inherit_cols: Columns that should be inherited by split
                tiles (e.g. K). Defaults to [].

        Returns:
            The grid with the null hypotheses added.
        """
        out = Grid(self.df.copy(), self.worker_id, copy.deepcopy(self.null_hypos))
        for H in null_hypos:
            out = out._add_null_hypo(H, inherit_cols)
            out.null_hypos.append(H)
        return out

    def _which_alternative(self):
        """
        Which tiles are in the alternative hypothesis space for all
        hypotheses.

        Returns:
            Boolean array of length n_tiles.
        """
        if len(self.null_hypos) == 0:
            return np.zeros(self.n_tiles, dtype=bool)
        null_truth = self.get_null_truth()
        return ~((null_truth.any(axis=1)) | (null_truth.shape[1] == 0))

    def prune_alternative(self):
        """
        Remove tiles that are not in the null hypothesis space for all
        hypotheses.
        Note that this method will not copy the grid if no tiles are pruned.

        Returns:
            The pruned grid.
        """
        if len(self.null_hypos) == 0:
            return self
        which = self._which_alternative()
        if not np.any(which):
            return self
        return self.subset(~which)

    def prune_inactive(self):
        """
        Get the active subset of the grid.

        Note that this method will not copy the grid if no tiles are pruned.

        Returns:
            A grid composed of only the active tiles.
        """
        if np.all(self.df["active"]):
            return self
        return self.subset(self.df["active"])

    def subset(self, which):
        """
        Subset a grid by some indexer.

        Args:
            which: The indexer.

        Returns:
            The grid subset.
        """
        df = self.df.loc[which].reset_index(drop=True)
        return Grid(df, self.worker_id, self.null_hypos)

    def add_cols(self, df):
        return Grid(pd.concat((self.df, df), axis=1), self.worker_id, self.null_hypos)

    def get_null_truth(self):
        return self.df[
            [
                f"null_truth{i}"
                for i in range(self.df.shape[1])
                if f"null_truth{i}" in self.df.columns
            ]
        ].to_numpy()

    def get_theta(self):
        return self.df[[f"theta{i}" for i in range(self.d)]].to_numpy()

    def get_radii(self):
        return self.df[[f"radii{i}" for i in range(self.d)]].to_numpy()

    def get_theta_and_vertices(self):
        theta = self.get_theta()
        return theta, (
            theta[:, None, :]
            + hypercube_vertices(self.d)[None, :, :] * self.get_radii()[:, None, :]
        )

    def refine(self, inherit_cols=[]):
        refine_radii = self.get_radii()[:, None, :] * 0.5
        refine_theta = self.get_theta()[:, None, :]
        new_thetas = (
            refine_theta + hypercube_vertices(self.d)[None, :, :] * refine_radii
        ).reshape((-1, self.d))
        new_radii = np.tile(refine_radii, (1, 2**self.d, 1)).reshape((-1, self.d))

        parent_id = np.repeat(self.df["id"].values, 2**self.d)
        out = _raw_init_grid(
            new_thetas,
            new_radii,
            self.worker_id,
            parents=parent_id,
        )
        _inherit(out.df, self.df, 2**self.d, inherit_cols)
        return out

    def repeat(self, n_reps):
        theta = np.repeat(self.get_theta(), n_reps, axis=0)
        radii = np.repeat(self.get_radii(), n_reps, axis=0)
        parents = np.repeat(self.df["id"].values, n_reps)
        return _raw_init_grid(theta, radii, self.worker_id, parents=parents)

    def concat(self, *others):
        return Grid(
            pd.concat((self.df, *[o.df for o in others]), axis=0, ignore_index=True),
            self.worker_id,
            self.null_hypos,
        )


def _inherit(child_df, parent_df, repeat, inherit_cols):
    assert (child_df["parent_id"] == np.repeat(parent_df["id"].values, repeat)).all()
    for col in inherit_cols:
        if col in child_df.columns:
            continue
        child_df[col] = np.repeat(parent_df[col].values, repeat)
    # NOTE: if we ever need a more complex parent-child relationship, we can
    # use pandas merge.
    # pd.merge(
    #     child_df,
    #     parent_df[['id'] + inherit_cols],
    #     left_on="parent_id",
    #     right_on="id",
    #     how='left',
    #     validate='many_to_one'
    # )


def _raw_init_grid(theta, radii, worker_id, parents=None):
    d = theta.shape[1]
    indict = dict()
    indict["id"] = _gen_short_uuids(theta.shape[0], worker_id=worker_id)

    # Is this a terminal tile in the tree?
    indict["active"] = True

    indict["parent_id"] = (
        parents.astype(np.uint64) if parents is not None else np.uint64(0)
    )

    for i in range(d):
        indict[f"theta{i}"] = theta[:, i]
    for i in range(d):
        indict[f"radii{i}"] = radii[:, i]

    return Grid(pd.DataFrame(indict), worker_id, [])


def create_grid(
    theta, *, radii=None, null_hypos=None, prune_alternative=True, prune_inactive=True
):
    """
    Create a grid from a set of points and radii.

    Args:
        theta: The parameter values for each tile.
        radii: The half-width for each tile. If None, we construct a Voronoi
            diagram of theta and use that to construct bounding boxes for each
            tile. Defaults to None.
        null_hypos: The null hypotheses to add. List of NullHypothesis objects.
        prune_alternative: Whether to prune the grid to only include tiles that
            are in the null hypothesis space.
        prune_inactive: Whether to prune the grid to only include active tiles.

    Returns:
        The grid.
    """
    if radii is None:
        # TODO: implement voronoi diagrim gridding.
        raise NotImplementedError("Voronoi gridding not implemented yet.")

    g = _raw_init_grid(theta, radii, 1)

    if null_hypos is not None:
        g = g.add_null_hypos(null_hypos)
    if prune_alternative:
        g = g.prune_alternative()
    if prune_inactive:
        g = g.prune_inactive()
    return g


def cartesian_grid(
    theta_min,
    theta_max,
    *,
    n=None,
    null_hypos=None,
    prune_alternative=True,
    prune_inactive=True,
):
    """
    Produce a grid of points in the hyperrectangle defined by theta_min and
    theta_max.

    Args:
        theta_min: The minimum value of theta for each dimension.
        theta_max: The maximum value of theta for each dimension.
        n: The number of theta values to use in each dimension.
        null_hypos: The null hypotheses to add. List of NullHypothesis objects.
        prune_alternative: Whether to prune the grid to only include tiles that
            are in the null hypothesis space.
        prune_inactive: Whether to prune the grid to only include active tiles.

    Returns:
        The grid.
    """
    theta_min = np.asarray(theta_min)
    theta_max = np.asarray(theta_max)

    if n is None:
        n = np.full(theta_min.shape[0], 2)
    theta, radii = _cartesian_gridpts(theta_min, theta_max, n)
    return create_grid(
        theta,
        radii=radii,
        null_hypos=null_hypos,
        prune_alternative=prune_alternative,
        prune_inactive=prune_inactive,
    )


def _cartesian_gridpts(theta_min, theta_max, n_theta_1d):
    theta_min = np.asarray(theta_min)
    theta_max = np.asarray(theta_max)
    n_theta_1d = np.asarray(n_theta_1d)

    n_arms = theta_min.shape[0]
    theta1d = [
        np.linspace(theta_min[i], theta_max[i], 2 * n_theta_1d[i] + 1)[1::2]
        for i in range(n_arms)
    ]
    radii1d = [
        np.full(
            theta1d[i].shape[0], (theta_max[i] - theta_min[i]) / (2 * n_theta_1d[i])
        )
        for i in range(n_arms)
    ]
    theta = np.stack(np.meshgrid(*theta1d), axis=-1).reshape((-1, len(theta1d)))
    radii = np.stack(np.meshgrid(*radii1d), axis=-1).reshape((-1, len(theta1d)))
    return theta, radii


def plot_grid(g: Grid, only_active=True, dims=(0, 1)):
    """
    Plot a 2D grid.

    Args:
        g: the grid
        null_hypos: If provided, the function will plot red lines for the null
            hypothesis boundaries. Defaults to [].
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from .planar_null import HyperPlane

    vertices = g.get_theta_and_vertices()[1][..., dims]

    if only_active:
        g = g.prune_inactive()

    polys = []
    for i in range(vertices.shape[0]):
        vs = vertices[i]
        vs = vs[~np.isnan(vs).any(axis=1)]
        centroid = np.mean(vs, axis=0)
        angles = np.arctan2(vs[:, 1] - centroid[1], vs[:, 0] - centroid[0])
        order = np.argsort(angles)
        polys.append(mpl.patches.Polygon(vs[order], fill=None, edgecolor="k"))
        plt.text(*centroid, str(i))

    plt.gca().add_collection(
        mpl.collections.PatchCollection(polys, match_original=True)
    )

    maxvs = np.max(vertices, axis=(0, 1))
    minvs = np.min(vertices, axis=(0, 1))
    view_center = 0.5 * (maxvs + minvs)
    view_radius = (maxvs - minvs) * 0.55
    xlims = view_center[0] + np.array([-1, 1]) * view_radius[0]
    ylims = view_center[1] + np.array([-1, 1]) * view_radius[1]
    plt.xlim(xlims)
    plt.ylim(ylims)

    for h in g.null_hypos:
        if not isinstance(h, HyperPlane):
            logger.warning("Skipping non-HyperPlane null hypothesis in plot_grid.")
            continue
        if h.n[0] == 0:
            xs = np.linspace(*xlims, 100)
            ys = (h.c - xs * h.n[0]) / h.n[1]
        else:
            ys = np.linspace(*ylims, 100)
            xs = (h.c - ys * h.n[1]) / h.n[0]
        plt.plot(xs, ys, "r-")


# https://stackoverflow.com/a/52229385/
def hypercube_vertices(d):
    """
    The corners of a hypercube of dimension d.

    >>> print(hypercube_vertices(1))
    [[-1]
     [ 1]]

    >>> print(hypercube_vertices(2))
    [[-1 -1]
     [-1  1]
     [ 1 -1]
     [ 1  1]]

    >>> print(hypercube_vertices(3))
    [[-1 -1 -1]
     [-1 -1  1]
     [-1  1 -1]
     [-1  1  1]
     [ 1 -1 -1]
     [ 1 -1  1]
     [ 1  1 -1]
     [ 1  1  1]]

    Args:
        d: the dimension

    Returns:
        a numpy array of shape (2**d, d) containing the vertices of the
        hypercube.
    """
    return np.array(list(product((-1, 1), repeat=d)))


def _get_edges(theta, radii):
    """
    Construct an array indicating the edges of each hyperrectangle.
    - edges[:, :, :n_params] are the vertices at the origin of the edges
    - edges[:, :, n_params:] are the edge vectors pointing from the start to
        the end of the edge

    Args:
        thetas: the centers of the hyperrectangles
        radii: the half-width of the hyperrectangles

    Returns:
        edges: an array as specified in the docstring shaped like
             (n_grid_pts, number of hypercube vertices, 2*n_params)
    """

    n_params = theta.shape[1]
    unit_vs = hypercube_vertices(n_params)
    n_vs = unit_vs.shape[0]
    unit_edges = []
    for i in range(n_vs):
        for j in range(n_params):
            if unit_vs[i, j] > 0:
                continue
            unit_edges.append(np.concatenate((unit_vs[i], np.identity(n_params)[j])))

    edges = np.tile(np.array(unit_edges)[None, :, :], (theta.shape[0], 1, 1))
    edges[:, :, :n_params] *= radii[:, None, :]
    edges[:, :, n_params:] *= 2 * radii[:, None, :]
    edges[:, :, :n_params] += theta[:, None, :]
    return edges


def _gen_short_uuids(n, worker_id, t=None):
    """
    Short UUIDs are a custom identifier created for imprint that should allow
    for concurrent creation of tiles without having overlapping indices.

    - The highest 28 bits are the time in seconds of creation. This will not
      loop for 8.5 years. When we start running jobs that take longer than 8.5
      years to complete, please send a message to me in the afterlife.
        - The unique_timer() function used for the time never returns the same
          time twice so the creation time is never re-used. If the creation
          time is going to be reused because less than one second has passed
          since the previous call to gen_short_uuids, then the timer increments
          by one.
    - The next 18 bits are the index of the process. This is a pretty generous limit
      on the number of processes. 2^18=262144.
    - The lowest 18 bits are the index of the created tiles within this batch.
      This allows for up to 2^18 = 262144 tiles to be created in a single
      batch. This is not a problematic constraint, because we can just
      increment the time by one and then grab another batch of IDs.

    NOTE: This should be safe across processes but will not be safe across
    threads within a single Python process because multithreaded programs share
    globals.

    Args:
        n: The number of short uuids to generate.
        worker_id: The host id. Must be > 0 and < 2**18. For non-concurrent
                   jobs, just set this to 1.
        t: The time to impose (used for testing). Defaults to None.

    Returns:
        An array with dtype uint64 of length n containing short uuids.
    """
    n_max = 2 ** _gen_short_uuids_one_batch.config[0] - 1
    if n <= n_max:
        return _gen_short_uuids_one_batch(n, worker_id, t)

    out = np.empty(n, dtype=np.uint64)
    for i in range(0, n, n_max):
        chunk_size = min(n_max, n - i)
        out[i : i + chunk_size] = _gen_short_uuids_one_batch(chunk_size, worker_id, t)
    return out


def _gen_short_uuids_one_batch(n, worker_id, t):
    n_bits, worker_bits = _gen_short_uuids_one_batch.config
    assert n < 2**n_bits

    assert worker_id > 0
    assert worker_id < 2**worker_bits

    if t is None:
        t = unique_timer()

    max_t = np.uint64(2 ** (64 - n_bits - worker_bits))
    looped_t = np.uint64(t) % max_t

    out = (
        (looped_t << np.uint64(n_bits + worker_bits))
        + (np.uint64(worker_id) << np.uint64(n_bits))
        + np.arange(n, dtype=np.uint64)
    )
    logger.debug(
        f"_gen_short_uuids(n={n}, worker_id={worker_id}, t={t}, n_bits={n_bits},"
        f" worker_bits={worker_bits}) = [{str(out[:3])[1:-1]}, ...]:"
    )
    return out


_gen_short_uuids_one_batch.config = (18, 18)
